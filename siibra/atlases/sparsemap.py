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

from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, TYPE_CHECKING, Union
import json
from pathlib import Path
import requests
import gzip
import hashlib

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from pandas import DataFrame
import nibabel as nib

from .parcellationmap import Map
from .region import Region
from ..retrieval.volume_fetcher import FetchKwargs, SIIBRA_MAX_FETCH_SIZE_GIB
from ..retrieval.file_fetcher.io.base import PartialReader
from ..attributes.locations import Point, PointCloud

if TYPE_CHECKING:
    from ..attributes.locations import BoundingBox


class SparseIndex:
    # TODO rename to siibrasparseindex
    HEADER = """SPARSEINDEX-UTF8-V0"""

    ALIAS_BBOX_SUFFIX = ".sparseindex.alias.json"
    PROBS_SUFFIX = ".sparseindex.probs.txt"
    VOXEL_SUFFIX = ".sparseindex.voxel.nii.gz"

    UINT32_MAX = 4_294_967_295

    readable = False
    writable = False

    def __new__(cls, *args, mode: Literal["r", "w"] = "r", **kwargs):
        if mode == "r":
            instance = object.__new__(ReadableSparseIndex)
            return instance
        if mode == "w":
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
                "You must call .add_img first, before the affine is populated."
            )
        return self._affine

    def _encode_regionname(self, regionname: str) -> str:
        return hashlib.md5(regionname.encode("utf-8")).hexdigest()[: self._aliassize]


class ReadableSparseIndex(SparseIndex):
    readable = True

    def __init__(self, filepath: Union[str, Path], **kwargs):
        super().__init__(filepath)

        self._readable_nii = None
        self._affine = None
        self._alias_dict: Union[None, Dict[str, Dict]] = None
        self._name_bbox_dict: Dict[str, Tuple[int, ...]] = {}

        if self.url:
            session = requests.Session()
            resp = session.get(self.url)
            header = resp.text
            assert header.startswith(
                self.HEADER
            ), f"header file does not start with {self.HEADER}, it was {header}."

            resp = session.get(self.url + self.VOXEL_SUFFIX)
            resp.raise_for_status()
            content = resp.content
            try:
                content = gzip.decompress(content)
            except gzip.BadGzipFile:
                ...
            self._readable_nii = nib.Nifti1Image.from_bytes(content)

            resp = session.get(self.url + self.ALIAS_BBOX_SUFFIX)
            resp.raise_for_status()
            self._alias_dict = resp.json()

        if self.filepath:
            self._readable_nii = nib.load(self.filepath.with_suffix(self.VOXEL_SUFFIX))
            with open(self.filepath.with_suffix(self.ALIAS_BBOX_SUFFIX), "r") as fp:
                self._alias_dict = json.load(fp=fp)

        assert self._alias_dict, "self._alias_dict not populated"
        for obj in self._alias_dict.values():
            name, bbox = obj.get("name"), obj.get("bbox")
            self._name_bbox_dict[name] = bbox

    @property
    def affine(self):
        if self._affine is None:
            self._affine = self._readable_nii.affine
        return self._affine

    def _decode_regionalias(self, alias: str) -> str:
        assert (
            alias in self._alias_dict
        ), f"alias={alias} not found in decoded dict {self._alias_dict}"
        return self._alias_dict[alias].get("name")

    def read(self, pos: Union[List[List[int]], np.ndarray]) -> Dict[str, float]:
        """
        For a given list of voxel coordinates, get the name and value mapped
        at each voxel.

        Parameters
        ----------
        pos : Union[List[List[int]], np.ndarray]

        Returns
        -------
        Dict[str, float]
            regionname: value
        """
        probreader = PartialReader(str(self.url or self.filepath) + self.PROBS_SUFFIX)
        probreader.open()

        pos = np.array(pos)
        assert (
            len(pos.shape) == 2 and pos.shape[1] == 3
        ), f"Expecting Nx3 array, but got {pos.shape}"

        nii = np.array(self._readable_nii.dataobj, dtype=np.uint64)
        x, y, z = pos.T

        result = []
        for val in nii[x, y, z].tolist():
            offset = val >> 32
            bytes_to_read = val & self.UINT32_MAX
            probreader.seek(int(offset))
            decoded = probreader.read(int(bytes_to_read))
            prob = {
                self._decode_regionalias(key): prob
                for key, prob in json.loads(decoded).items()
            }
            result.append(prob)

        probreader.close()
        return result

    def get_boundingbox_extrema(self, regionname: str, **kwargs) -> List[int]:
        assert (
            regionname in self._name_bbox_dict
        ), f"regionname={regionname} not found in self._alias_name_dict={self._alias_dict}"

        return self._name_bbox_dict[regionname]


@dataclass(repr=False, eq=False)
class SparseMap(Map):

    @property
    def _readable_sparseindex(self) -> ReadableSparseIndex:
        pass

    def fetch(
        self,
        region: Union[str, Region] = None,
        frmt: str = None,
        bbox: "BoundingBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        if region is None:
            assert len(self.regions) == 1, ValueError(
                "To fetch a volume from a SparseMap, please provide a region "
                "name mapped in this SparseMap."
            )
            matched = self.regions[0]
        else:
            if isinstance(region, Region):
                matched = region.name
            else:
                matched = self.parcellation.get_region(region).name

        assert (
            matched in self.regions
        ), f"Statistical map of region '{matched}' is not available in '{self.name}'."

        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
        )
        return super().fetch(region=matched, frmt=frmt, **fetch_kwargs)

    def designate_points(
        self,
        points: Union[Point, PointCloud],
        **fetch_kwargs: FetchKwargs,
    ) -> DataFrame:
        if self._readable_sparseindex is None:
            return super().designate_points(points, **fetch_kwargs)

        points_ = (
            PointCloud.from_points([points]) if isinstance(points, Point) else points
        )
        if any(s not in {0.0} for s in points_.sigma):
            raise ValueError(
                f"Cannot designate uncertain points. Please use '{self.get_intersection_score.__name__}' instead."
            )

        points_wrpd = points_.warp(self.space_id)

        first_volume = self.find_volumes(self.regions[0])[0]
        # TODO: consider just using affine to transform the points
        vx, vy, vz = first_volume._points_to_voxels_coords(points_wrpd)

        assignments: List[Map.RegionAssignment] = []
        for pointindex, region, map_value in enumerate(
            zip(*self._readable_sparseindex.read(np.stack(vx, vy, vz)))
        ):
            if map_value == 0:
                continue
            assignments.append(
                Map.RegionAssignment(
                    input_structure_index=pointindex,
                    centroid=points_[pointindex].coordinate,
                    map_value=map_value,
                    region=region,
                )
            )
        return Map._convert_point_samples_to_dataframe(assignments)
