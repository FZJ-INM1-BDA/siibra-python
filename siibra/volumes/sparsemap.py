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

from .parcellationmap import Map

from ..commons import MapIndex, logger
from ..locations import BoundingBox
from ..retrieval import CACHE

from os import path
import gzip
from typing import Dict
from nilearn import image
from nibabel import Nifti1Image, load
from tqdm import tqdm
import numpy as np


class SparseIndex:

    def __init__(self):
        self.probs = []
        self.bboxes = []

        # these are initialized when adding the first volume, see below
        self.affine: np.ndarray = None
        self.shape = None
        self.voxels: np.ndarray = None

    def add_img(self, img: Nifti1Image):

        if self.num_volumes == 0:
            self.affine = img.affine
            self.shape = img.shape
            self.voxels = np.zeros(img.shape, dtype=np.int32) - 1
        else:
            assert img.shape == self.shape
            assert (img.affine - self.affine).sum() == 0

        volume = self.num_volumes
        imgdata = np.asanyarray(img.dataobj)
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

    def to_cache(self, prefix: str):
        """ Serialize this index to the cache,
        using the given prefix for the cache filenames. """
        probsfile = CACHE.build_filename(f"{prefix}", suffix="probs.txt.gz")
        bboxfile = CACHE.build_filename(f"{prefix}", suffix="bboxes.txt.gz")
        voxelfile = CACHE.build_filename(f"{prefix}", suffix="voxels.nii.gz")
        Nifti1Image(self.voxels, self.affine).to_filename(voxelfile)
        with gzip.open(probsfile, 'wt') as f:
            for D in self.probs:
                f.write("{}\n".format(" ".join(f"{i} {p}" for i, p in D.items())))
        with gzip.open(bboxfile, "wt") as f:
            for bbox in self.bboxes:
                f.write(
                    "{} {}\n".format(
                        " ".join(map(str, bbox["minpoint"])),
                        " ".join(map(str, bbox["maxpoint"])),
                    )
                )

    @classmethod
    def from_cache(cls, prefix: str):
        """
        Attemts to build a sparse index from the siibra cache,
        looking for suitable cache files with the specified prefix.
        Returns None if cached files are not found or suitable.
        """

        probsfile = CACHE.build_filename(f"{prefix}", suffix="probs.txt.gz")
        bboxfile = CACHE.build_filename(f"{prefix}", suffix="bboxes.txt.gz")
        voxelfile = CACHE.build_filename(f"{prefix}", suffix="voxels.nii.gz")
        if not all(path.isfile(f) for f in [probsfile, bboxfile, voxelfile]):
            return None

        result = SparseIndex()

        voxels = load(voxelfile)
        result.voxels = np.asanyarray(voxels.dataobj)
        result.affine = voxels.affine
        result.shape = voxels.shape

        with gzip.open(probsfile, "rt") as f:
            lines = f.readlines()
            for line in tqdm(
                lines,
                total=len(lines),
                desc="Loading sparse index",
                unit="voxels",
                disable=logger.level > 20,
            ):
                fields = line.strip().split(" ")
                mapindices = list(map(int, fields[0::2]))
                values = list(map(float, fields[1::2]))
                D = dict(zip(mapindices, values))
                result.probs.append(D)

        with gzip.open(bboxfile, "rt") as f:
            for line in f:
                fields = line.strip().split(" ")
                result.bboxes.append(
                    {
                        "minpoint": tuple(map(int, fields[:3])),
                        "maxpoint": tuple(map(int, fields[3:])),
                    }
                )

        return result


class SparseMap(Map):
    """A sparse representation of list of continuous (e.g. probabilistic) brain region maps.

    It represents the 3D continuous maps of N brain regions by two data structures:
        1) 'spatial_index', a 3D volume where non-negative values represent unique
            indices into a list of region assignments
        2) 'probs', a list of region assignments where each entry is a dict

    More precisely, given
        i = sparse_index.voxels[x, y, z]
    we define that
        - if i<0, no brain region is assigned at this location
        - if i>=0, probs[i] defines the probabilities of brain regions.

    Each entry in probs is a dictionary that represents the region assignments for
    the unique voxel where spatial_index==i. The assignment maps from a "mapindex"
    to the actual (probability) value.
    """

    # A gitlab instance with holds precomputed sparse indices
    _GITLAB_SERVER = 'https://jugit.fz-juelich.de'
    _GITLAB_PROJECT = 5779

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
        datasets: list = [],
    ):
        Map.__init__(
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
    def sparse_index(self):
        if self._sparse_index_cached is None:
            prefix = f"{self.parcellation.id}_{self.space.id}_{self.maptype}_index"
            spind = SparseIndex.from_cache(prefix)
            if spind is None:
                spind = SparseIndex()
                for volume in tqdm(
                    range(len(self)), total=len(self), unit="maps",
                    desc=f"Fetching {len(self)} volumetric maps",
                    disable=logger.level > 20,
                ):
                    img = super().fetch(volume=volume)
                    if img is None:
                        region = self.get_region(volume=volume)
                        logger.error(f"Cannot retrieve volume #{volume} for {region.name}, it will not be included in the sparse map.")
                        continue
                    spind.add_img(img)
                spind.to_cache(prefix)
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
        volume: int = None,
        resolution_mm: float = None,
        voi: BoundingBox = None,
        variant: str = None,
        format: str = None,
        index: MapIndex = None,
        cropped: bool = False,
    ):
        """
        Recreate a particular volumetric map from the sparse
        representation.

        Arguments
        ---------
        volume : int
            The index of the mapped volume to be fetched.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes.
            If None, the smallest possible resolution will be chosen.
            If -1, the largest feasible resolution will be chosen.
        voi: VolumeOfInterest
            bounding box specification
        variant : str
            Optional specification of a specific variant to use for the maps. For example,
            fsaverage provides the 'pial', 'white matter' and 'inflated' surface variants.
        format: str
            optional specificatino of the voume format to use (e.g. "nii", "neuroglancer/precomputed")
        cropped: Boolean
            If true, only a cropped image of the nonzero values with
            appropriate affine matrix is returned, otherwise a full-sized
            volume with padded zeros (Default: False)

        """
        if index is not None:
            assert volume is None
            assert isinstance(index, MapIndex)
            volume = index.volume
        elif isinstance(volume, MapIndex):
            # be kind if an index is passed as the first parameter
            volume = volume.volume
        elif isinstance(volume, str):
            # be kind if a region name is passed as the first parameter
            logger.info(
                f"'{volume}' will be interpreted as a region name to decode the volume index."
            )
            index = self.get_index(volume)
            volume = index.volume
        if voi is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support volume of interest fetching yet."
            )
        if resolution_mm is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support fetching at resolutions other than 1mm yet."
            )
        if volume is None:
            assert index is not None
            volume = index.volume
        assert isinstance(volume, int)

        x, y, z, v = self.sparse_index.mapped_voxels(volume)
        if cropped:
            bbox = np.array([[min(_), max(_)] for _ in [x, y, z]])
            result = np.zeros(bbox[:, 1] - bbox[:, 0] + 1)
            x0, y0, z0 = bbox[:, 0]
            result[x - x0, y - y0, z - z0] = v
            shift = np.identity(4)
            shift[:3, -1] = bbox[:, 0]
            return Nifti1Image(result, np.dot(self.affine, shift))
        else:
            result = np.zeros(self.shape, dtype=np.float32)
            result[x, y, z] = v
            return Nifti1Image(result, self.affine)

    def _read_voxel(self, x, y, z):
        spind = self.sparse_index
        vx = spind.voxels[x, y, z]
        if isinstance(vx, int):
            return list(
                (None, volume, value)
                for volume, value in spind.probs[vx].items()
            )
        else:
            return list(
                (pointindex, volume, value)
                for pointindex, voxel in enumerate(vx)
                for volume, value in spind.probs[voxel].items()
            )

    def _assign_image(self, queryimg: Nifti1Image, minsize_voxel: int, lower_threshold: float):
        """
        Assign an image volume to this parcellation map.

        Parameters:
        -----------
        queryimg: Nifti1Image
            the image to be compared with maps
        minsize_voxel: int, default: 1
            Minimum voxel size of image components to be taken into account.
        lower_threshold: float, default: 0
            Lower threshold on values in the continuous map. Values smaller than
            this threshold will be excluded from the assignment computation.
        """

        assignments = []
        components = None

        # resample query image into this image's voxel space, if required
        if (queryimg.affine - self.affine).sum() == 0:
            queryimg = queryimg
        else:
            if issubclass(np.asanyarray(queryimg.dataobj).dtype.type, np.integer):
                interp = "nearest"
            else:
                interp = "linear"
            queryimg = image.resample_img(
                queryimg,
                target_affine=self.affine,
                target_shape=self.shape,
                interpolation=interp,
            )

        querydata = np.asanyarray(queryimg.dataobj).squeeze()

        for mode, modeimg in Map.iterate_connected_components(queryimg):

            # determine bounding box of the mode
            modemask = np.asanyarray(modeimg.dataobj)
            XYZ2 = np.array(np.where(modemask)).T
            if XYZ2.shape[0] <= minsize_voxel:
                components[modemask] == 0
                continue
            X2, Y2, Z2 = [v.squeeze() for v in np.split(XYZ2, 3, axis=1)]

            bbox2 = BoundingBox(XYZ2.min(0), XYZ2.max(0) + 1, space=None)
            if bbox2.volume == 0:
                continue

            spind = self.sparse_index

            for volume in tqdm(
                range(len(self)),
                desc=f"Assigning to {len(self)} sparse maps",
                total=len(self),
                unit=" map",
                disable=logger.level > 20,
            ):
                bbox1 = BoundingBox(
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
                contains = intersection / (v1 > 0).sum()
                contained = intersection / (v2 > 0).sum()

                v1d = v1 - v1.mean()
                v2d = v2 - v2.mean()
                rho = (
                    (v1d * v2d).sum()
                    / np.sqrt((v1d ** 2).sum())
                    / np.sqrt((v2d ** 2).sum())
                )

                maxval = v1.max()

                assignments.append(
                    [mode, volume, maxval, iou, contained, contains, rho]
                )

        return assignments
