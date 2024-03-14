# Copyright 2018-2022
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
"""Represents lists of probabilistic brain region maps."""
from . import parcellationmap, volume as _volume

from .providers import provider
from ..commons import MapIndex, logger, connected_components, siibra_tqdm
from ..locations import boundingbox
from ..retrieval.cache import CACHE
from ..retrieval.requests import HttpRequest, FileLoader
from ..exceptions import (
    InsufficientArgumentException, ExcessiveArgumentException
)

from os import path, makedirs
from typing import Dict, Union, TYPE_CHECKING, List
from nilearn import image
import numpy as np

if TYPE_CHECKING:
    from ..core.region import Region


class SparseIndex:

    # Precomputed sparse indices are stored in an EBRAINS data proxy
    _DATAPROXY_BASEURL = "https://data-proxy.ebrains.eu/api/v1/buckets/reference-atlas-data/sparse-indices/"

    _SUFFIXES = {
        "probs": ".sparseindex.probs.txt.gz",
        "bboxes": ".sparseindex.bboxes.txt.gz",
        "voxels": ".sparseindex.voxels.nii.gz"
    }

    def __init__(self):
        self.probs = []
        self.bboxes = []

        # these are initialized when adding the first volume, see below
        self.affine: np.ndarray = None
        self.shape = None
        self.voxels: np.ndarray = None

    def add_img(self, imgdata: np.ndarray, affine: np.ndarray):

        if self.num_volumes == 0:
            self.affine = affine
            self.shape = imgdata.shape
            self.voxels = np.zeros(imgdata.shape, dtype=np.int32) - 1
        else:
            if (imgdata.shape != self.shape) or ((affine - self.affine).sum() != 0):
                raise RuntimeError(
                    "Building sparse maps from volumes with different voxel spaces is not yet supported in siibra."
                )

        volume = self.num_volumes
        X, Y, Z = [v.astype("int32") for v in np.where(imgdata > 0)]
        for x, y, z, prob in zip(X, Y, Z, imgdata[X, Y, Z]):
            coord_id = self.voxels[x, y, z]
            if coord_id >= 0:
                # Coordinate already seen. Add observed value.
                assert volume not in self.probs[coord_id]
                assert len(self.probs) > coord_id
                self.probs[coord_id][volume] = prob
            else:
                # New coordinate. Append entry with observed value.
                coord_id = len(self.probs)
                self.voxels[x, y, z] = coord_id
                self.probs.append({volume: prob})

        self.bboxes.append(
            {
                "minpoint": (X.min(), Y.min(), Z.min()),
                "maxpoint": (X.max(), Y.max(), Z.max()),
            }
        )

    @property
    def num_volumes(self):
        return len(self.bboxes)

    def max(self):
        return self.voxels.max()

    def coords(self, volume: int):
        # Nx3 array with x/y/z coordinates of the N nonzero values of the given mapindex
        assert volume in range(self.num_volumes)
        coord_ids = [i for i, l in enumerate(self.probs) if volume in l]
        x0, y0, z0 = self.bboxes[volume]["minpoint"]
        x1, y1, z1 = self.bboxes[volume]["maxpoint"]
        return (
            np.array(
                np.where(
                    np.isin(
                        self.voxels[x0: x1 + 1, y0: y1 + 1, z0: z1 + 1],
                        coord_ids,
                    )
                )
            ).T
            + (x0, y0, z0)
        ).T

    def mapped_voxels(self, volume: int):
        # returns the x, y, and z coordinates of nonzero voxels for the map
        # with the given index, together with their corresponding values v.
        assert volume in range(self.num_volumes)
        x, y, z = [v.squeeze() for v in np.split(self.coords(volume), 3)]
        v = [self.probs[i][volume] for i in self.voxels[x, y, z]]
        return x, y, z, v

    @classmethod
    def load(cls, filepath_or_url: str) -> "SparseIndex":
        """
        Loads a precomputed SparseIndex to the memory.

        Parameters
        ----------
        filepath_or_url: str
            Path/url to the SparseIndex files
            (eg. https://url_to_files/basefilename):
            - basefilename.sparseindex.probs.txt.gz
            - basefilename.sparseindex.bboxes.txt.gz
            - basefilename.sparseindex.voxels.nii.gz

        Returns
        -------
        SparseIndex
        """
        from gzip import decompress
        spindtxt_decoder = lambda b: decompress(b).decode('utf-8').strip().splitlines()

        probsfile = filepath_or_url + SparseIndex._SUFFIXES["probs"]
        bboxfile = filepath_or_url + SparseIndex._SUFFIXES["bboxes"]
        voxelfile = filepath_or_url + SparseIndex._SUFFIXES["voxels"]
        if all(path.isfile(f) for f in [probsfile, bboxfile, voxelfile]):
            request = FileLoader
        else:
            request = HttpRequest

        result = cls.__init__()

        voxels = request(voxelfile).get()
        result.voxels = np.asanyarray(voxels.dataobj)
        result.affine = voxels.affine
        result.shape = voxels.shape

        lines_probs = request(probsfile, func=spindtxt_decoder).get()
        for line in siibra_tqdm(
            lines_probs,
            total=len(lines_probs),
            desc="Loading sparse index",
            unit="voxels"
        ):
            fields = line.strip().split(" ")
            mapindices = list(map(int, fields[0::2]))
            values = list(map(float, fields[1::2]))
            D = dict(zip(mapindices, values))
            result.probs.append(D)

        lines_bboxes = request(bboxfile, func=spindtxt_decoder).get()
        for line in lines_bboxes:
            fields = line.strip().split(" ")
            result.bboxes.append({
                "minpoint": tuple(map(int, fields[:3])),
                "maxpoint": tuple(map(int, fields[3:])),
            })

        return result

    def save(self, base_filename: str, folder: str = ""):
        """
        Save SparseIndex (3x) files to under the folder `folder`
        with base_filename. If SparseIndex is not cached, siibra will first
        create it first.

        Parameters
        ----------
        base_filename: str
            The files that will be created as:
            - base_filename.sparseindex.probs.txt.gz
            - base_filename.sparseindex.bboxes.txt.gz
            - base_filename.sparseindex.voxels.nii.gz

        folder: str, default=""
        """
        from nibabel import Nifti1Image
        import gzip

        fullpath = path.join(folder, base_filename)
        logger.info(f"Saving SparseIndex to '{base_filename}' with suffixes {SparseIndex._SUFFIXES}")

        if folder and not path.isdir(folder):
            makedirs(folder)

        Nifti1Image(self.voxels, self.affine).to_filename(
            fullpath + SparseIndex._SUFFIXES["voxels"]
        )
        with gzip.open(fullpath + SparseIndex._SUFFIXES["probs"], 'wt') as f:
            for D in self.probs:
                f.write(
                    "{}\n".format(
                        " ".join(f"{i} {p}" for i, p in D.items())
                    )
                )
        with gzip.open(fullpath + SparseIndex._SUFFIXES["bboxes"], "wt") as f:
            for bbox in self.bboxes:
                f.write(
                    "{} {}\n".format(
                        " ".join(map(str, bbox["minpoint"])),
                        " ".join(map(str, bbox["maxpoint"])),
                    )
                )
        logger.info(f"SparseIndex is saved to {fullpath}.")

    @classmethod
    def from_sparsemap(cls, sparsemap: "SparseMap") -> "SparseIndex":
        with provider.SubvolumeProvider.UseCaching():
            spind = cls.__init__()
            for img in siibra_tqdm(
                sparsemap.fetch_iter(), total=len(sparsemap), unit="maps",
                desc="Fetching volumetric maps and computing SparseIndex"
            ):
                spind.add_img(np.asanyarray(img.dataobj), img.affine)
        return spind


class SparseMap(parcellationmap.Map):
    """
    A sparse representation of list of statistical (e.g. probabilistic) brain
    region maps.

    It represents the 3D statistical maps of N brain regions by two data structures:

    1) 'spatial_index', a 3D volume where non-negative values represent unique indices into a list of region assignments
    2) 'probs', a list of region assignments where each entry is a dict

    More precisely, given ``i = sparse_index.voxels[x, y, z]`` we define that

    - if `i<0`, no brain region is assigned at this location
    - if `i>=0`, ``probs[i]`` defines the probabilities of brain regions.

    Each entry in probs is a dictionary that represents the region assignments for
    the unique voxel where ``spatial_index == i``. The assignment maps from a MapIndex
    to the actual (probability) value.
    """

    def __init__(
        self,
        identifier: str,
        name: str,
        space_spec: dict,
        parcellation_spec: dict,
        indices: Dict[str, MapIndex],
        volumes: list = [],
        shortname: str = "",
        description: str = "",
        modality: str = None,
        publications: list = [],
        datasets: list = []
    ):
        parcellationmap.Map.__init__(
            self,
            identifier=identifier,
            name=name,
            space_spec=space_spec,
            parcellation_spec=parcellation_spec,
            indices=indices,
            shortname=shortname,
            description=description,
            modality=modality,
            publications=publications,
            datasets=datasets,
            volumes=volumes,
        )
        self._sparse_index_cached = None

    @property
    def _cache_prefix(self):
        return CACHE.build_filename(f"{self.parcellation.id}_{self.space.id}_{self.maptype}_{self.name}_index")

    @property
    def sparse_index(self):
        if self._sparse_index_cached is None:
            try:  # try loading from cache on disk
                spind = SparseIndex.load(self._cache_prefix)
            except Exception:
                spind = None
            if spind is None:  # try from precomputed source
                try:
                    logger.info("Downloading and loading precomputed SparseIndex...")
                    fname = f"{self.name.replace(' ', '_').replace('statistical', 'continuous')}"
                    spind = SparseIndex.load(SparseIndex._DATAPROXY_BASEURL + fname)
                except Exception:
                    logger.error("Failed to download precomputed SparseIndex.", exc_info=1)
            if spind is None:  # Download each map and compute the SparseIndex
                spind = SparseIndex.from_sparsemap(self)
                spind.save(self._cache_prefix, folder=CACHE.folder)
            self._sparse_index_cached = spind
        assert self._sparse_index_cached.max() == len(self._sparse_index_cached.probs) - 1
        return self._sparse_index_cached

    @property
    def affine(self):
        return self.sparse_index.affine

    @property
    def shape(self):
        return self.sparse_index.shape

    def fetch(
        self,
        region_or_index: Union[MapIndex, str, 'Region'] = None,
        *,
        index: MapIndex = None,
        region: Union[str, 'Region'] = None,
        **kwargs
    ):
        """
        Recreate a particular volumetric map from the sparse
        representation.

        Parameters
        ----------
        region_or_index: str, Region, MapIndex
            Lazy match the specification.
        index : MapIndex
            The index to be fetched.
        region: str, Region
            Region name specification. If given, will be used to decode the map
            index of a particular region.

        Returns
        -------
        An image or mesh
        """
        if kwargs.get('format') in ['mesh'] + _volume.Volume.MESH_FORMATS:
            # a mesh is requested, this is not handled by the sparse map
            return super().fetch(region_or_index, index=index, region=region, **kwargs)

        try:
            length = len([arg for arg in [region_or_index, region, index] if arg is not None])
            assert length == 1
        except AssertionError:
            if length > 1:
                raise ExcessiveArgumentException(
                    "One and only one of region_or_index, region, index can be defined for fetch"
                )
            # user can provide no arguments, which assumes one and only one volume present

        if isinstance(region_or_index, MapIndex):
            index = region_or_index

        from ..core.region import Region
        if isinstance(region_or_index, (str, Region)):
            region = region_or_index

        volidx = None
        if index is not None:
            assert isinstance(index, MapIndex)
            volidx = index.volume
        if region is not None:
            index = self.get_index(region)
            assert index is not None
            volidx = index.volume

        if volidx is None:
            try:
                assert len(self) == 1
                volidx = 0
            except AssertionError:
                raise InsufficientArgumentException(
                    f"{self.__class__.__name__} provides {len(self)} volumes. "
                    "Specify 'region' or 'index' for fetch() to identify one."
                )

        assert isinstance(volidx, int)
        x, y, z, v = self.sparse_index.mapped_voxels(volidx)
        result = np.zeros(self.shape, dtype=np.float32)
        result[x, y, z] = v
        volume = _volume.from_array(
            data=result,
            affine=self.affine,
            space=self.space,
            name=f"Sparse map of {region} from {self.parcellation} in {self.space}"
        )
        return volume.fetch()

    def _read_voxel(self, x, y, z):
        spind = self.sparse_index
        vx = spind.voxels[x, y, z]
        if isinstance(vx, int):
            return list(
                (None, volume, None, value)
                for volume, value in spind.probs[vx].items()
            )
        else:
            return list(
                (pointindex, volume, None, value)
                for pointindex, voxel in enumerate(vx)
                for volume, value in spind.probs[voxel].items()
            )

    def _assign_volume(
        self,
        queryvolume: "_volume.Volume",
        minsize_voxel: int,
        lower_threshold: float,
        split_components: bool = True
    ) -> List[parcellationmap.AssignImageResult]:
        """
        Assign an image volume to this sparse map.

        Parameters:
        -----------
        queryvolume: Volume
            the volume to be compared with maps
        minsize_voxel: int, default: 1
            Minimum voxel size of image components to be taken into account.
        lower_threshold: float, default: 0
            Lower threshold on values in the statistical map. Values smaller than
            this threshold will be excluded from the assignment computation.
        split_components: bool, default: True
            Whether to split the query volume into disjoint components.
        """
        queryimg = queryvolume.fetch()
        imgdata = np.asanyarray(queryimg.dataobj)
        imgaffine = queryimg.affine
        assignments = []

        # resample query image into this image's voxel space, if required
        if (imgaffine - self.affine).sum() == 0:
            querydata = imgdata.squeeze()
        else:
            if issubclass(imgdata.dtype.type, np.integer):
                interp = "nearest"
            else:
                interp = "linear"
            from nibabel import Nifti1Image
            queryimg = image.resample_img(
                Nifti1Image(imgdata, imgaffine),
                target_affine=self.affine,
                target_shape=self.shape,
                interpolation=interp,
            )
            querydata = np.asanyarray(queryimg.dataobj).squeeze()

        iter_func = connected_components if split_components \
            else lambda img: [(1, img)]

        for mode, modemask in iter_func(querydata):

            # determine bounding box of the mode
            XYZ2 = np.array(np.where(modemask)).T
            position = np.dot(self.affine, np.r_[XYZ2.mean(0), 1])[:3]
            if XYZ2.shape[0] <= minsize_voxel:
                continue
            X2, Y2, Z2 = [v.squeeze() for v in np.split(XYZ2, 3, axis=1)]

            bbox2 = boundingbox.BoundingBox(XYZ2.min(0), XYZ2.max(0) + 1, space=None)
            if bbox2.volume == 0:
                continue

            spind = self.sparse_index

            for volume in siibra_tqdm(
                range(len(self)),
                desc=f"Assigning structure #{mode} to {len(self)} sparse maps",
                total=len(self),
                unit=" map"
            ):
                bbox1 = boundingbox.BoundingBox(
                    self.sparse_index.bboxes[volume]["minpoint"],
                    self.sparse_index.bboxes[volume]["maxpoint"],
                    space=None,
                )
                if bbox1.intersection(bbox2) is None:
                    continue

                # compute union of voxel space bounding boxes
                bbox = bbox1.union(bbox2)
                bbshape = np.array(bbox.shape, dtype="int") + 1
                x0, y0, z0 = map(int, bbox.minpoint)

                # build flattened vector of map values
                v1 = np.zeros(np.prod(bbshape))
                XYZ1 = spind.coords(volume).T
                X1, Y1, Z1 = [v.squeeze() for v in np.split(XYZ1, 3, axis=1)]
                indices1 = np.ravel_multi_index(
                    (X1 - x0, Y1 - y0, Z1 - z0), bbshape
                )
                v1[indices1] = [spind.probs[i][volume] for i in spind.voxels[X1, Y1, Z1]]
                v1[v1 < lower_threshold] = 0

                # build flattened vector of input image mode
                v2 = np.zeros(np.prod(bbshape))
                indices2 = np.ravel_multi_index(
                    (X2 - x0, Y2 - y0, Z2 - z0), bbshape
                )
                v2[indices2] = querydata[X2, Y2, Z2]

                assert v1.shape == v2.shape

                intersection = np.sum(
                    (v1 > 0) & (v2 > 0)
                )  # np.minimum(v1, v2).sum()
                if intersection == 0:
                    continue
                iou = intersection / np.sum(
                    (v1 > 0) | (v2 > 0)
                )  # np.maximum(v1, v2).sum()

                v1d = v1 - v1.mean()
                v2d = v2 - v2.mean()
                rho = (
                    (v1d * v2d).sum()
                    / np.sqrt((v1d ** 2).sum())
                    / np.sqrt((v2d ** 2).sum())
                )

                maxval = v1.max()

                assignments.append(
                    parcellationmap.AssignImageResult(
                        input_structure=mode,
                        centroid=tuple(position.round(2)),
                        volume=volume,
                        fragment=None,
                        map_value=maxval,
                        intersection_over_union=iou,
                        intersection_over_first=intersection / (v1 > 0).sum(),
                        intersection_over_second=intersection / (v2 > 0).sum(),
                        correlation=rho,
                        weighted_mean_of_first=np.sum(v1 * v2) / np.sum(v2),
                        weighted_mean_of_second=np.sum(v1 * v2) / np.sum(v1)
                    )
                )

        return assignments
