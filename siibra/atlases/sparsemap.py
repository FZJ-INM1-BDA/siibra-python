# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, replace, asdict
from typing import List, Dict, Tuple, Union, TYPE_CHECKING, Iterable
import json
from pathlib import Path
import requests
import gzip
import hashlib
from io import BytesIO

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from pandas import DataFrame
import nibabel as nib
import os

from .parcellationmap import Map
from .region import Region
from ..attributes.dataproviders.volume.image import ImageProvider
from ..attributes.dataproviders.volume import VolumeOpsKwargs, SIIBRA_MAX_FETCH_SIZE_GIB
from ..attributes.locations import Point, PointCloud, BoundingBox
from ..attributes.locations.boundingbox import (
    from_imageprovider as bbox_from_imageprovider,
)
from ..operations.file_fetcher.io.base import PartialReader
from ..operations.file_fetcher.io import MemoryPartialReader
from ..commons.logger import siibra_tqdm, logger, QUIET
from ..commons.conf import KEEP_LOCAL_CACHE, MEMORY_HUNGRY
from ..cache import CACHE
from ..operations.image_assignment import (
    ImageAssignment,
    ScoredImageAssignment,
    get_intersection_scores,
)


SPARSEINDEX_BASEURL = (
    "https://data-proxy.ebrains.eu/api/v1/buckets/reference-atlas-data/sparse-indices/"
)


class SparseIndex:
    # TODO rename to siibrasparseindex
    HEADER = """SPARSEINDEX-UTF8-V0"""

    ALIAS_BBOX_SUFFIX = ".sparseindex.alias.json"
    PROBS_SUFFIX = ".sparseindex.probs.txt"
    VOXEL_SUFFIX = ".sparseindex.voxel.nii.gz"

    UINT32_MAX = 4_294_967_295

    readable = False
    writable = False

    def __new__(cls, *args, mode: Literal["r", "w"] = "r", inmemory=False, **kwargs):
        if mode == "r":
            if inmemory:
                instance = object.__new__(InMemoryReadableSparseIndex)
                return instance
            else:
                instance = object.__new__(ReadableSparseIndex)
                return instance
        if mode == "w":
            if inmemory:
                raise RuntimeError("inmemory cannot be used with write mode")
            instance = object.__new__(WritableSparseIndex)
            return instance
        return super().__new__(cls)

    def __init__(self, filepath: Union[str, Path], mode: Literal["r", "w"] = "r"):
        self.filepath = None
        self.url = None
        self.mode = mode

        if isinstance(filepath, str) and filepath.startswith("https://"):
            self.url = filepath
            return
        self.filepath = Path(filepath)

    def read(self, pos: Union[List[List[int]], np.ndarray]):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def add_img(self, nii: nib.Nifti1Image, regionname: str):
        raise NotImplementedError

    def get_img(self, regionname: str) -> nib.Nifti1Image:
        raise NotImplementedError

    def get_boundingbox_extrema(self, regionname: str, **kwargs) -> List[int]:
        raise NotImplementedError

    @property
    def affine(self):
        raise NotImplementedError


class WritableSparseIndex(SparseIndex):
    writable = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.filepath is None:
            raise RuntimeError("WritableSparseIndex must point to a local file")

        self._region_name_mapping: Dict[str, str] = {}
        """regionname -> regionalias"""

        self._probs: List[Dict[str, float]] = []
        """regionalias -> prob"""

        self._bbox: Dict[str, List[int]] = {}
        """regionalias -> bbox"""

        self._voxels: Dict[Tuple[int, int, int], int] = {}
        """voxelcoord -> element index in self._probs"""

        self._shape = None
        self._affine = None
        self._aliassize = 6

    def save(self):
        if self._affine is None:
            raise RuntimeError("No image has been added yet")

        basename = self.filepath
        assert basename, "self.filepath is not defined. It must be defined."
        basename.parent.mkdir(parents=True, exist_ok=True)

        # newline-separated json objects on all region metadata
        regionmetas = {}
        for regionname, regionalias in self._region_name_mapping.items():
            assert (
                regionalias in self._bbox
            ), f"Expected regionalias {regionalias} in _bbox, but was not. regionname={regionname}"
            regionmetas[regionalias] = {
                "name": regionname,
                "bbox": self._bbox[regionalias],
            }

        # write metadata
        with open(basename.with_suffix(self.ALIAS_BBOX_SUFFIX), "w") as fp:
            json.dump(regionmetas, indent=2, fp=fp)

        current_offset = 0

        offset_record: List[Tuple[int, int]] = []
        """offset/bytes"""

        with open(basename.with_suffix(self.PROBS_SUFFIX), "w") as fp:
            for prob in self._probs:
                str_to_write = json.dumps(prob) + "\n"
                byte_count = len(str_to_write.encode("utf-8"))
                fp.write(str_to_write)

                # purely for ascethic and easy debugging purpose

                offset_record.append((current_offset, byte_count))
                current_offset += byte_count

        # saving voxel map, containing offset
        lut = np.zeros(self._shape, dtype=np.uint64, order="C")
        for (x, y, z), list_idx in self._voxels.items():
            offset, bytes_used = offset_record[list_idx]
            assert offset < self.UINT32_MAX, "offset > unit32 max"
            lut[x, y, z] = np.uint64(offset << 32) + np.uint64(bytes_used)

        nii = nib.Nifti1Image(lut, affine=self._affine, dtype=np.uint64)
        nii.to_filename(basename.with_suffix(self.VOXEL_SUFFIX))

        # saving root meta file
        with open(basename, "w") as fp:
            fp.write(self.HEADER)

    def add_img(self, nii: nib.Nifti1Image, regionname: str):
        if self._shape is None:
            self._shape = nii.dataobj.shape
        else:
            assert np.all(
                self._shape == nii.dataobj.shape
            ), "Sparse index from different shape not supported is None"

        if self._affine is None:
            self._affine = nii.affine
        else:
            assert np.all(
                self._affine == nii.affine
            ), "Sparse index from different affine is not supported"

        assert (
            regionname not in self._region_name_mapping
        ), f"{regionname} has already been mapped"

        regionalias = self._encode_regionname(regionname)
        self._region_name_mapping[regionname] = regionalias

        assert (
            regionalias not in self._bbox
        ), f"regionalias={regionalias} already added to _bbox regionname={regionname}"

        assert (
            regionalias not in self._probs
        ), f"regionalias={regionalias} already added to _probs regionname={regionname}"

        imgdata = np.asanyarray(nii.dataobj)
        X, Y, Z = [v.astype("int32") for v in np.where(imgdata > 0)]
        for x, y, z, prob in zip(X, Y, Z, imgdata[X, Y, Z]):
            coord_id = (x, y, z)
            prob = prob.astype("float")

            if coord_id not in self._voxels:
                self._probs.append({regionalias: prob})
                self._voxels[coord_id] = len(self._probs) - 1

            if coord_id in self._voxels:
                probidx = self._voxels[coord_id]
                self._probs[probidx][regionalias] = prob

        self._bbox[regionalias] = np.array(
            [X.min(), Y.min(), Z.min(), X.max(), Y.max(), Z.max()]
        ).tolist()

    @property
    def affine(self):
        if self._affine is None:
            raise RuntimeError(
                f"You must call `{self.__class__.__name__}.add_img` first, before the affine is populated."
            )
        return self._affine

    def _encode_regionname(self, regionname: str) -> str:
        return hashlib.md5(regionname.encode("utf-8")).hexdigest()[: self._aliassize]


class ReadableSparseIndex(SparseIndex):
    readable = True

    def __init__(self, filepath: Union[str, Path], **kwargs):
        super().__init__(filepath)

        self._header = None
        self._affine = None
        self._dataobj = None
        self._alias_dict: Union[None, Dict[str, Dict]] = None
        self._name_bbox_dict: Dict[str, Tuple[int, ...]] = {}

        _readable_nii: Union[nib.Nifti1Image, None] = None
        if self.url:
            session = requests.Session()
            resp = session.get(self.url)
            self._header = resp.text

            resp = session.get(self.url + self.VOXEL_SUFFIX)
            resp.raise_for_status()
            content = resp.content
            try:
                content = gzip.decompress(content)
            except gzip.BadGzipFile:
                ...
            _readable_nii = nib.Nifti1Image.from_bytes(content)

            resp = session.get(self.url + self.ALIAS_BBOX_SUFFIX)
            resp.raise_for_status()
            self._alias_dict = resp.json()

        if self.filepath:
            with open(self.filepath, "r") as fp:
                self._header = fp.read()
            _readable_nii = nib.load(self.filepath.with_suffix(self.VOXEL_SUFFIX))
            with open(self.filepath.with_suffix(self.ALIAS_BBOX_SUFFIX), "r") as fp:
                self._alias_dict = json.load(fp=fp)

        assert _readable_nii is not None
        self._affine = _readable_nii.affine
        self._dataobj = np.array(_readable_nii.dataobj)
        assert self._dataobj.dtype == np.uint64

        assert self._header.startswith(
            self.HEADER
        ), f"header file does not start with {self.HEADER}, it was {self._header}."
        assert self._alias_dict, "self._alias_dict not populated"
        for obj in self._alias_dict.values():
            name, bbox = obj.get("name"), obj.get("bbox")
            self._name_bbox_dict[name] = bbox

    @property
    def affine(self):
        return self._affine

    @property
    def dataobj(self):
        return self._dataobj

    def _decode_regionalias(self, alias: str) -> str:
        assert (
            alias in self._alias_dict
        ), f"alias={alias} not found in decoded dict {self._alias_dict}"
        return self._alias_dict[alias].get("name")

    def _read(
        self, reader: PartialReader, pos: Union[List[List[int]], np.ndarray]
    ) -> List[Dict[str, float]]:

        # standardise user input into ndarray
        pos = np.array(pos)
        assert (
            len(pos.shape) == 2 and pos.shape[1] == 3
        ), f"Expecting Nx3 array, but got {pos.shape}"

        x, y, z = pos.T

        result = []
        for val in self.dataobj[x, y, z].tolist():
            offset = val >> 32
            bytes_to_read = val & self.UINT32_MAX
            reader.seek(int(offset))
            decoded = reader.read(int(bytes_to_read))
            prob = {
                self._decode_regionalias(key): prob
                for key, prob in json.loads(decoded).items()
            }
            result.append(prob)
        return result

    def read(self, pos: Union[List[List[int]], np.ndarray]):
        """
        For a given list of voxel coordinates, get the name and value mapped
        at each voxel.

        Parameters
        ----------
        pos: Union[List[List[int]], np.ndarray]
            expects an NX3 array

        Returns
        -------
        List[Dict[str, float]]
            list of regionname: value
        """

        # TODO make partial readers into contextmanagers
        probreader = PartialReader(str(self.url or self.filepath) + self.PROBS_SUFFIX)
        probreader.open()
        result = self._read(probreader, pos)
        probreader.close()
        return result

    def get_img(self, regionname: str) -> nib.Nifti1Image:
        minx, miny, minz, maxx, maxy, maxz = self.get_boundingbox_extrema(regionname)
        wanted_dataarray = np.zeros(self.dataobj.shape, dtype=np.float32)
        # np.arange return [start, end), thus + 1 on max end
        X, Y, Z = (
            np.arange(minx, maxx + 1),
            np.arange(miny, maxy + 1),
            np.arange(minz, maxz + 1),
        )
        coords = np.mgrid[
            minx : maxx + 1,
            miny : maxy + 1,
            minz : maxz + 1,
        ]

        wanted_dataarray[X, Y, Z] = [
            float(v.get(regionname, 0)) for v in self.read(coords)
        ]
        return nib.Nifti1Image(wanted_dataarray, affine=self.affine)

    def get_boundingbox_extrema(self, regionname: str, **kwargs) -> List[int]:
        """Returns Tuple consisting of 6 int, xmin, ymin, zmin, xmax, ymax, zmax, representing the voxel indicies
        where the non-zero value of the statistical"""
        assert (
            regionname in self._name_bbox_dict
        ), f"regionname={regionname} not found in self._alias_name_dict={self._alias_dict}"

        return self._name_bbox_dict[regionname]

    def _iter_file(self) -> Iterable[Tuple[str, BytesIO]]:
        yield self.ALIAS_BBOX_SUFFIX, BytesIO(
            json.dumps(self._alias_dict).encode("utf-8")
        )

        nii = nib.Nifti1Image(self.dataobj, affine=self.affine, dtype=np.uint64)
        yield self.VOXEL_SUFFIX, gzip.compress(nii.to_bytes())

        reader = PartialReader(str(self.url or self.filepath) + self.PROBS_SUFFIX)
        reader.open()
        yield self.PROBS_SUFFIX, reader
        reader.close()

        # save the meta file at the very end. The presence of this file indicates that the rest of the file saved correctly
        yield "", BytesIO(self._header.encode("utf-8"))

    def save_as(self, filepath: str):
        basename = Path(filepath)
        CHUNK_SIZE = 1024 * 1024 * 4  # 4mb chunk size
        for suffix, bio in self._iter_file():

            with open(basename.with_suffix(suffix), "wb") as fp:
                if isinstance(bio, bytes):
                    fp.write(bio)
                    continue

                offset = 0
                while True:
                    bio.seek(offset)
                    _bytes = bio.read(CHUNK_SIZE)
                    fp.write(_bytes)
                    if len(_bytes) < CHUNK_SIZE:
                        break
                    offset += CHUNK_SIZE


class InMemoryReadableSparseIndex(ReadableSparseIndex):
    def __init__(self, filepath: Union[str, Path], **kwargs):
        super().__init__(filepath, **kwargs)
        self._memreader: MemoryPartialReader
        if self.url:
            resp = requests.get(self.url + self.PROBS_SUFFIX)
            resp.raise_for_status()
            self._memreader = MemoryPartialReader(resp.content)
        if self.filepath:
            path = Path(self.filepath)
            with open(path.with_suffix(self.PROBS_SUFFIX), "rb") as fp:
                self._memreader = MemoryPartialReader(fp.read())
        assert self._memreader is not None

    def read(self, pos: Union[List[List[int]], np.ndarray]) -> List[Dict[str, float]]:
        return self._read(self._memreader, pos)


@dataclass(repr=False, eq=False)
class SparseMap(Map):

    def _get_readable_sparseindex(
        self, warmup=False, inmemory=False
    ) -> ReadableSparseIndex:
        sparseindicies = [
            attr for attr in self.attributes if attr.schema == "x-siibra/sparseindex"
        ]
        if len(sparseindicies) != 1:
            logger.debug(
                f"Expected one and only one sparse index volume, but got {len(sparseindicies)}"
            )
            return None
        spidx = sparseindicies[0]
        url = spidx.extra.get("url")

        localcache = CACHE.build_filename(url)
        if not url:
            logger.warning(f"SparseIndex volume deformed. {spidx.__dict__}")
            return None

        if Path(localcache).is_file():
            return SparseIndex(localcache, mode="r", inmemory=inmemory)

        if warmup:
            spidx = SparseIndex(url, mode="r", inmemory=inmemory)
            logger.info(f"Caching sparseindex at {url}")
            spidx.save_as(localcache)
            return SparseIndex(localcache, mode="r", inmemory=inmemory)

        return SparseIndex(url, mode="r")

    def _save_sparseindex(self, filepath):
        wspind = SparseIndex(filepath, mode="w")
        for regionname in siibra_tqdm(self.regionnames, unit="region"):
            extracted = self.extract_regional_map(regionname)
            wspind.add_img(nii=extracted.get_data(), regionname=regionname)
        wspind.save()

    def lookup_points(
        self,
        points: Union[Point, PointCloud],
        **fetch_kwargs: VolumeOpsKwargs,
    ) -> DataFrame:
        spind = self._get_readable_sparseindex(
            warmup=KEEP_LOCAL_CACHE > 0, inmemory=MEMORY_HUNGRY > 0
        )
        if spind is None:
            return super().lookup_points(points, **fetch_kwargs)

        points_ = (
            PointCloud.from_points([points]) if isinstance(points, Point) else points
        )
        if any(s not in {0.0} for s in points_.sigma):
            logger.warning(
                f"To get the full asignment score of uncertain points please use `{self.assign.__name__}`."
                "`lookup_points()` only considers the voxels the coordinates correspond to."
            )
            points_ = replace(points_, sigma=np.zeros(len(points_)).tolist())

        points_wrpd = points_.warp(self.space_id)

        volume_provider = self._extract_regional_map_volume_provider(
            self.regionnames[0]
        )
        # TODO: consider just using affine to transform the points
        vx, vy, vz = volume_provider._points_to_voxels_coords(points_wrpd)

        assignments: List[Map.RegionAssignment] = []
        for pointindex, readout in enumerate(spind.read(np.stack([vx, vy, vz]).T)):
            for regionname, map_value in readout.items():
                if map_value == 0:
                    continue
                assignments.append(
                    Map.RegionAssignment(
                        input_structure_index=pointindex,
                        centroid=points_[pointindex].coordinate,
                        map_value=map_value,
                        regionname=regionname,
                    )
                )
        return Map._convert_assignments_to_dataframe(assignments)

    def warmup(self):
        self._get_readable_sparseindex(warmup=True)

    def assign(
        self,
        queryitem: Union[Point, PointCloud, ImageProvider],
        split_components: bool = True,
        voxel_sigma_threshold: int = 3,
        iou_lower_threshold=0,
        statistical_map_lower_threshold: float = 0,
        **volume_ops_kwargs: VolumeOpsKwargs,
    ) -> DataFrame:
        from ..attributes.locations.ops.intersection import intersect

        # TODO implement properly, allow for map_value population
        logger.warning("SparseMap.assign has not yet been implemented correctly")

        if isinstance(queryitem, Point) and queryitem.sigma == 0:
            return self.lookup_points(queryitem, **volume_ops_kwargs)
        if isinstance(queryitem, PointCloud):
            sigmas = set(queryitem.sigma)
            if len(sigmas) == 1 and 0 in sigmas:
                return self.lookup_points(queryitem, **volume_ops_kwargs)

        spind = self._get_readable_sparseindex()
        queryitemloc = (
            bbox_from_imageprovider(queryitem)
            if isinstance(queryitem, ImageProvider)
            else queryitem
        )
        assignments = []
        for regionname in self.regionnames:
            xmin, ymin, zmin, xmax, ymax, zmax = spind.get_boundingbox_extrema(
                regionname
            )
            bbox = BoundingBox(minpoint=[xmin, ymin, zmin], maxpoint=[xmax, ymax, zmax])
            bbox = bbox.transform(spind.affine, space_id=self.space_id)
            if not intersect(bbox, queryitemloc):
                continue
            region_image = self._extract_regional_map_volume_provider(
                regionname=regionname, frmt="image", **volume_ops_kwargs
            )
            for assgnmt in get_intersection_scores(
                queryitem=queryitem,
                target_image=region_image,
                split_components=split_components,
                voxel_sigma_threshold=voxel_sigma_threshold,
                iou_lower_threshold=iou_lower_threshold,
                target_masking_lower_threshold=statistical_map_lower_threshold,
                **volume_ops_kwargs,
            ):
                if isinstance(assgnmt, ScoredImageAssignment):
                    assignments.append(
                        Map.ScoredRegionAssignment(
                            **asdict(assgnmt),
                            regionname=regionname,
                        )
                    )
                else:
                    assignments.append(
                        Map.RegionAssignment(
                            **asdict(assgnmt),
                            regionname=regionname,
                        )
                    )
        return self._convert_assignments_to_dataframe(assignments)
