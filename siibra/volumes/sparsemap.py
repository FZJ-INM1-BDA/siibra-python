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

from .map import Map
from .util import create_gaussian_kernel

from ..commons import MapIndex, logger
from ..core.location import BoundingBox, Point, PointSet
from ..retrieval import CACHE

from os import path
import gzip
from typing import Dict, Union
from nilearn import image
from nibabel import Nifti1Image, load
from tqdm import tqdm
import numpy as np
from numbers import Number
import pandas as pd


class SparseIndex:

    def __init__(self):
        self.probs = []
        self.bboxes = []

        # these are initialized when adding the first volume, see below
        self.affine: np.ndarray = None
        self.shape = None
        self.voxels: Nifti1Image = None

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
        ebrains_ids: dict = {},
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
            ebrains_ids=ebrains_ids,
            volumes=volumes,
        )

        self._sparse_index_cached = None

    @property
    def sparse_index(self):
        if self._sparse_index_cached is None:
            prefix = f"{self.parcellation.id}_{self.space.id}_{self.maptype}_index"
            spind = SparseIndex.from_cache(prefix)
            if spind is None:
                print("index not in cache")
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

    def assign(
        self,
        item: Union[Point, PointSet, Nifti1Image],
        msg=None,
        quiet=False,
        minsize_voxel=1,
        lower_threshold=0.0,
        skip_mapindices=[],
    ):
        """Assign an input image to brain regions.

        The input image is assumed to be defined in the same coordinate space
        as this parcellation map.

        Parameters
        ----------
        item: Point, PointSet, or Nifti1Image
            A spatial object defined in the same physical reference space as this
            parcellation map, which could be a point, set of points, or image.
            If it is an image, it will be resampled to the same voxel space if its affine
            transforation differs from that of the parcellation map.
            Resampling will use linear interpolation for float image types,
            otherwise nearest neighbor.
        msg: str, or None
            An optional message to be shown with the progress bar. This is
            useful if you use assign() in a loop.
        quiet: Bool, default: False
            If True, no outputs will be generated.
        minsize_voxel: int, default: 1
            Minimum voxel size of image components to be taken into account.
        lower_threshold: float, default: 0
            Lower threshold on values in the continuous map. Values smaller than
            this threshold will be excluded from the assignment computation.
        skip_mapindices: list, default: []
            Maps whose index is listed here will not be considered for the assignment

        Return
        ------
        assignments : pandas Dataframe
            A table of associated regions and their scores per component found in the input image,
            or per coordinate provived.
            The scores are:
                - MaxValue: Maximum value of the voxels in the map covered by an input coordinate or
                  input image signal component.
                - Pearson correlation coefficient between the brain region map and an input image signal
                  component (NaN for exact coordinates)
                - "Contains": Percentage of the brain region map contained in an input image signal component,
                  measured from their binarized masks as the ratio between the volume of their interesection
                  and the volume of the brain region (NaN for exact coordinates)
                - "Contained"": Percentage of an input image signal component contained in the brain region map,
                  measured from their binary masks as the ratio between the volume of their interesection
                  and the volume of the input image signal component (NaN for exact coordinates)
        components: Nifti1Image, or None
            If the input was an image, this is a labelled volume mapping the detected components
            in the input image, where pixel values correspond to the "component" column of the
            assignment table. If the input was a Point or PointSet, this is None.
        """

        assignments = []
        components = None

        if isinstance(item, Point):
            item = PointSet([item], item.space, sigma_mm=item.sigma)

        if isinstance(item, PointSet):
            if item.space != self.space:
                logger.info(
                    f"Coordinates will be converted from {item.space.name} "
                    f"to {self.space.name} space for assignment."
                )
            # convert sigma to voxel coordinates
            scaling = np.array(
                [np.linalg.norm(self.affine[:, i]) for i in range(3)]
            ).mean()
            phys2vox = np.linalg.inv(self.affine)

            for pointindex, point in enumerate(item.warp(self.space)):

                sigma_vox = point.sigma / scaling
                if sigma_vox < 3:
                    # voxel-precise - just read out the value in the maps
                    N = len(self)
                    logger.info(f"Assigning coordinate {tuple(point)} to {N} maps")
                    x, y, z = (np.dot(phys2vox, point.homogeneous) + 0.5).astype("int")[
                        :3
                    ]
                    for mapindex, value in self.sparse_index.probs[
                        self.sparse_index.voxels[x, y, z]
                    ].items():
                        if mapindex in skip_mapindices:
                            continue
                        if value > lower_threshold:
                            assignments.append(
                                (
                                    pointindex,
                                    mapindex,
                                    value,
                                    np.NaN,
                                    np.NaN,
                                    np.NaN,
                                    np.NaN,
                                )
                            )
                else:
                    logger.info(
                        f"Assigning uncertain coordinate {tuple(point)} to {len(self)} maps."
                    )
                    kernel = create_gaussian_kernel(sigma_vox, 3)
                    r = int(kernel.shape[0] / 2)  # effective radius
                    xyz_vox = (np.dot(phys2vox, point.homogeneous) + 0.5).astype("int")
                    shift = np.identity(4)
                    shift[:3, -1] = xyz_vox[:3] - r
                    # build niftiimage with the Gaussian blob,
                    # then recurse into this method with the image input
                    W = Nifti1Image(dataobj=kernel, affine=np.dot(self.affine, shift))
                    T, _ = self.assign(
                        W,
                        lower_threshold=lower_threshold,
                        skip_mapindices=skip_mapindices,
                    )
                    assignments.extend(
                        [
                            [
                                pointindex,
                                mapindex,
                                maxval,
                                iou,
                                contained,
                                contains,
                                rho,
                            ]
                            for (
                                _,
                                mapindex,
                                _,
                                maxval,
                                rho,
                                iou,
                                contains,
                                contained,
                            ) in T.values
                        ]
                    )

        elif isinstance(item, Nifti1Image):

            # ensure query image is in parcellation map's voxel space
            if (item.affine - self.affine).sum() == 0:
                img2 = item
            else:
                if issubclass(np.asanyarray(item.dataobj).dtype.type, np.integer):
                    interp = "nearest"
                else:
                    interp = "linear"
                img2 = image.resample_img(
                    item,
                    target_affine=self.affine,
                    target_shape=self.shape,
                    interpolation=interp,
                )
            img2data = np.asanyarray(img2.dataobj).squeeze()

            # split input image into multiple 'modes',  ie. connected components
            from skimage import measure

            components = measure.label(img2data > 0)
            component_labels = np.unique(components)
            assert component_labels[0] == 0
            if len(component_labels) > 1:
                logger.info(
                    f"Detected {len(component_labels)-1} components in the image. Assigning each of them to {len(self)} brain regions."
                )

            for modeindex in component_labels[1:]:

                # determine bounding box of the mode
                mask = components == modeindex
                XYZ2 = np.array(np.where(mask)).T
                if XYZ2.shape[0] <= minsize_voxel:
                    components[mask] == 0
                    continue
                X2, Y2, Z2 = [v.squeeze() for v in np.split(XYZ2, 3, axis=1)]

                bbox2 = BoundingBox(XYZ2.min(0), XYZ2.max(0) + 1, space=None)
                if bbox2.volume == 0:
                    continue

                for mapindex in tqdm(
                    range(len(self)),
                    total=len(self),
                    unit=" map",
                    desc=msg,
                    disable=logger.level > 20,
                ):
                    if mapindex in skip_mapindices:
                        continue

                    bbox1 = BoundingBox(
                        self.sparse_index.bboxes[mapindex]["minpoint"],
                        self.sparse_index.bboxes[mapindex]["maxpoint"],
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
                    XYZ1 = self._coords(mapindex).T
                    X1, Y1, Z1 = [v.squeeze() for v in np.split(XYZ1, 3, axis=1)]
                    indices1 = np.ravel_multi_index(
                        (X1 - x0, Y1 - y0, Z1 - z0), bbshape
                    )
                    v1[indices1] = [
                        self.sparse_index.probs[i][mapindex] for i in self.sparse_index.voxels[X1, Y1, Z1]
                    ]
                    v1[v1 < lower_threshold] = 0

                    # build flattened vector of input image mode
                    v2 = np.zeros(np.prod(bbshape))
                    indices2 = np.ravel_multi_index(
                        (X2 - x0, Y2 - y0, Z2 - z0), bbshape
                    )
                    v2[indices2] = img2data[X2, Y2, Z2]

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
                        [modeindex, mapindex, maxval, iou, contained, contains, rho]
                    )

        else:
            raise RuntimeError(
                f"Items of type {item.__class__.__name__} cannot be used for region assignment."
            )

        if len(assignments) == 0:
            df = pd.DataFrame(
                columns=[
                    "Component",
                    "MapIndex",
                    "Region",
                    "MaxValue",
                    "Correlation",
                    "IoU",
                    "Contains",
                    "Contained",
                ]
            )
        else:
            result = np.array(assignments)
            # sort by component, then by correlation
            ind = np.lexsort((-result[:, -1], result[:, 0]))

            df = pd.DataFrame(
                {
                    "Component": result[ind, 0].astype("int"),
                    "MapIndex": result[ind, 1].astype("int"),
                    "Region": [
                        self.decode_index(mapindex=m, labelindex=None).name
                        for m in result[ind, 1]
                    ],
                    "MaxValue": result[ind, 2],
                    "Correlation": result[ind, 6],
                    "IoU": result[ind, 3],
                    "Contains": result[ind, 5],
                    "Contained": result[ind, 4],
                }
            ).dropna(axis=1, how="all")

        if components is None:
            return df
        else:
            return df, Nifti1Image(components, self.affine)

    def assign(
        self,
        item: Union[Point, PointSet, Nifti1Image],
        msg=None,
        quiet=False,
        minsize_voxel=1,
        lower_threshold=0.0,
        skip_mapindices=[],
    ):
        """Assign an input image to brain regions.

        The input image is assumed to be defined in the same coordinate space
        as this parcellation map.

        Parameters
        ----------
        item: Point, PointSet, or Nifti1Image
            A spatial object defined in the same physical reference space as this
            parcellation map, which could be a point, set of points, or image.
            If it is an image, it will be resampled to the same voxel space if its affine
            transforation differs from that of the parcellation map.
            Resampling will use linear interpolation for float image types,
            otherwise nearest neighbor.
        msg: str, or None
            An optional message to be shown with the progress bar. This is
            useful if you use assign() in a loop.
        quiet: Bool, default: False
            If True, no outputs will be generated.
        minsize_voxel: int, default: 1
            Minimum voxel size of image components to be taken into account.
        lower_threshold: float, default: 0
            Lower threshold on values in the continuous map. Values smaller than
            this threshold will be excluded from the assignment computation.
        skip_mapindices: list, default: []
            Maps whose index is listed here will not be considered for the assignment

        Return
        ------
        assignments : pandas Dataframe
            A table of associated regions and their scores per component found in the input image,
            or per coordinate provived.
            The scores are:
                - MaxValue: Maximum value of the voxels in the map covered by an input coordinate or
                  input image signal component.
                - Pearson correlation coefficient between the brain region map and an input image signal
                  component (NaN for exact coordinates)
                - "Contains": Percentage of the brain region map contained in an input image signal component,
                  measured from their binarized masks as the ratio between the volume of their interesection
                  and the volume of the brain region (NaN for exact coordinates)
                - "Contained"": Percentage of an input image signal component contained in the brain region map,
                  measured from their binary masks as the ratio between the volume of their interesection
                  and the volume of the input image signal component (NaN for exact coordinates)
        components: Nifti1Image, or None
            If the input was an image, this is a labelled volume mapping the detected components
            in the input image, where pixel values correspond to the "component" column of the
            assignment table. If the input was a Point or PointSet, this is None.
        """

        assignments = []
        components = None

        if isinstance(item, Point):
            item = PointSet([item], item.space, sigma_mm=item.sigma)

        if isinstance(item, PointSet):
            if item.space != self.space:
                logger.info(
                    f"Coordinates will be converted from {item.space.name} "
                    f"to {self.space.name} space for assignment."
                )
            # convert sigma to voxel coordinates
            scaling = np.array(
                [np.linalg.norm(self.affine[:, i]) for i in range(3)]
            ).mean()
            phys2vox = np.linalg.inv(self.affine)

            for pointindex, point in enumerate(item.warp(self.space.id)):

                sigma_vox = point.sigma / scaling
                if sigma_vox < 3:
                    # voxel-precise - just read out the value in the maps
                    N = len(self)
                    logger.info(f"Assigning coordinate {tuple(point)} to {N} maps")
                    x, y, z = (np.dot(phys2vox, point.homogeneous) + 0.5).astype("int")[
                        :3
                    ]
                    for mapindex, value in self.sparse_index.probs[
                        self.sparse_index.voxels[x, y, z]
                    ].items():
                        if mapindex in skip_mapindices:
                            continue
                        if value > lower_threshold:
                            assignments.append(
                                (
                                    pointindex,
                                    mapindex,
                                    value,
                                    np.NaN,
                                    np.NaN,
                                    np.NaN,
                                    np.NaN,
                                )
                            )
                else:
                    logger.info(
                        f"Assigning uncertain coordinate {tuple(point)} to {len(self)} maps."
                    )
                    kernel = create_gaussian_kernel(sigma_vox, 3)
                    r = int(kernel.shape[0] / 2)  # effective radius
                    xyz_vox = (np.dot(phys2vox, point.homogeneous) + 0.5).astype("int")
                    shift = np.identity(4)
                    shift[:3, -1] = xyz_vox[:3] - r
                    # build niftiimage with the Gaussian blob,
                    # then recurse into this method with the image input
                    W = Nifti1Image(dataobj=kernel, affine=np.dot(self.affine, shift))
                    T, _ = self.assign(
                        W,
                        lower_threshold=lower_threshold,
                        skip_mapindices=skip_mapindices,
                    )
                    assignments.extend(
                        [
                            [
                                pointindex,
                                mapindex,
                                maxval,
                                iou,
                                contained,
                                contains,
                                rho,
                            ]
                            for (
                                _,
                                mapindex,
                                _,
                                maxval,
                                rho,
                                iou,
                                contains,
                                contained,
                            ) in T.values
                        ]
                    )

        elif isinstance(item, Nifti1Image):

            # ensure query image is in parcellation map's voxel space
            if (item.affine - self.affine).sum() == 0:
                img2 = item
            else:
                if issubclass(np.asanyarray(item.dataobj).dtype.type, np.integer):
                    interp = "nearest"
                else:
                    interp = "linear"
                img2 = image.resample_img(
                    item,
                    target_affine=self.affine,
                    target_shape=self.shape,
                    interpolation=interp,
                )
            img2data = np.asanyarray(img2.dataobj).squeeze()

            # split input image into multiple 'modes',  ie. connected components
            from skimage import measure

            components = measure.label(img2data > 0)
            component_labels = np.unique(components)
            assert component_labels[0] == 0
            if len(component_labels) > 1:
                logger.info(
                    f"Detected {len(component_labels)-1} components in the image. Assigning each of them to {len(self)} brain regions."
                )

            for modeindex in component_labels[1:]:

                # determine bounding box of the mode
                mask = components == modeindex
                XYZ2 = np.array(np.where(mask)).T
                if XYZ2.shape[0] <= minsize_voxel:
                    components[mask] == 0
                    continue
                X2, Y2, Z2 = [v.squeeze() for v in np.split(XYZ2, 3, axis=1)]

                bbox2 = BoundingBox(XYZ2.min(0), XYZ2.max(0) + 1, space=None)
                if bbox2.volume == 0:
                    continue

                for mapindex in tqdm(
                    range(len(self)),
                    total=len(self),
                    unit=" map",
                    desc=msg,
                    disable=logger.level > 20,
                ):
                    if mapindex in skip_mapindices:
                        continue

                    bbox1 = BoundingBox(
                        self.bboxes[mapindex]["minpoint"],
                        self.bboxes[mapindex]["maxpoint"],
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
                    XYZ1 = self._coords(mapindex).T
                    X1, Y1, Z1 = [v.squeeze() for v in np.split(XYZ1, 3, axis=1)]
                    indices1 = np.ravel_multi_index(
                        (X1 - x0, Y1 - y0, Z1 - z0), bbshape
                    )
                    v1[indices1] = [
                        self.probs[i][mapindex] for i in self.spatial_index[X1, Y1, Z1]
                    ]
                    v1[v1 < lower_threshold] = 0

                    # build flattened vector of input image mode
                    v2 = np.zeros(np.prod(bbshape))
                    indices2 = np.ravel_multi_index(
                        (X2 - x0, Y2 - y0, Z2 - z0), bbshape
                    )
                    v2[indices2] = img2data[X2, Y2, Z2]

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
                        [modeindex, mapindex, maxval, iou, contained, contains, rho]
                    )

        else:
            raise RuntimeError(
                f"Items of type {item.__class__.__name__} cannot be used for region assignment."
            )

        if len(assignments) == 0:
            df = pd.DataFrame(
                columns=[
                    "Component",
                    "MapIndex",
                    "Region",
                    "MaxValue",
                    "Correlation",
                    "IoU",
                    "Contains",
                    "Contained",
                ]
            )
        else:
            result = np.array(assignments)
            # sort by component, then by correlation
            ind = np.lexsort((-result[:, -1], result[:, 0]))

            df = pd.DataFrame(
                {
                    "Component": result[ind, 0].astype("int"),
                    "MapIndex": result[ind, 1].astype("int"),
                    "Region": [
                        self.get_region(volume=m, label=None).name
                        for m in result[ind, 1]
                    ],
                    "MaxValue": result[ind, 2],
                    "Correlation": result[ind, 6],
                    "IoU": result[ind, 3],
                    "Contains": result[ind, 5],
                    "Contained": result[ind, 4],
                }
            ).dropna(axis=1, how="all")

        if components is None:
            return df
        else:
            return df, Nifti1Image(components, self.affine)