# Copyright 2018-2025
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
"""A specific mesh or 3D array."""

from typing import List, Dict, Union, Set, TYPE_CHECKING
from dataclasses import dataclass
from time import sleep
import json
from functools import lru_cache

import numpy as np
from nibabel import Nifti1Image
from skimage import feature as skimage_feature, filters

from . import providers as _providers
from ..commons import (
    resample_img_to_img,
    siibra_tqdm,
    affine_scaling,
    connected_components,
    logger,
    QUIET,
)
from ..exceptions import NoMapAvailableError, SpaceWarpingFailedError, EmptyPointCloudError
from ..retrieval import requests
from ..core import space as _space, structure
from ..core.concept import get_registry
from ..locations import point, pointcloud, boundingbox

if TYPE_CHECKING:
    from ..retrieval.datasets import EbrainsDataset as TypeDataset


@dataclass
class ComponentSpatialProperties:
    """
    Centroid and nonzero volume of an image.
    """
    centroid: point.Point
    volume: int

    @staticmethod
    def compute_from_image(
        img: Nifti1Image,
        space: Union[str, "_space.Space"],
        split_components: bool = True

    ) -> List["ComponentSpatialProperties"]:
        """
        Find the center of an image in its (non-zero) voxel space and and its
        volume.

        Parameters
        ----------
        img: Nifti1Image
        space: str, Space
        split_components: bool, default: True
            If True, finds the spatial properties for each connected component
            found by skimage.measure.label.
        """
        scale = affine_scaling(img.affine)
        if split_components:
            iter_components = lambda img: connected_components(
                np.asanyarray(img.dataobj),
                connectivity=None
            )
        else:
            iter_components = lambda img: [(0, np.asanyarray(img.dataobj))]

        spatial_props: List[ComponentSpatialProperties] = []
        for _, component in iter_components(img):
            nonzero: np.ndarray = np.c_[np.nonzero(component)]
            spatial_props.append(
                ComponentSpatialProperties(
                    centroid=point.Point(
                        np.dot(img.affine, np.r_[nonzero.mean(0), 1])[:3],
                        space=space
                    ),
                    volume=nonzero.shape[0] * scale,
                )
            )

        # sort by volume
        spatial_props.sort(key=lambda cmp: cmp.volume, reverse=True)

        return spatial_props


class Volume(structure.BrainStructure):
    """
    A volume is a specific mesh or 3D array,
    which can be accessible via multiple providers in different formats.
    """

    IMAGE_FORMATS = [
        "nii",
        "zip/nii",
        "neuroglancer/precomputed"
    ]

    MESH_FORMATS = [
        "neuroglancer/precompmesh",
        "neuroglancer/precompmesh/surface",
        "gii-mesh",
        "gii-label",
        "freesurfer-annot",
        "zip/freesurfer-annot",
    ]

    SUPPORTED_FORMATS = IMAGE_FORMATS + MESH_FORMATS

    _FORMAT_LOOKUP = {
        "image": IMAGE_FORMATS,
        "mesh": MESH_FORMATS,
        "surface": MESH_FORMATS,
        "nifti": ["nii", "zip/nii"],
        "nii": ["nii", "zip/nii"]
    }

    _FETCH_CACHE = {}  # we keep a cache of the most recently fetched volumes
    _FETCH_CACHE_MAX_ENTRIES = 3

    def __init__(
        self,
        space_spec: dict,
        providers: List[_providers.provider.VolumeProvider],
        name: str = "",
        variant: str = None,
        datasets: List['TypeDataset'] = [],
        bbox: "boundingbox.BoundingBox" = None
    ):
        self._name = name
        self._space_spec = space_spec
        self.variant = variant
        self._providers: Dict[str, _providers.provider.VolumeProvider] = {}
        self.datasets = datasets
        self._boundingbox = bbox
        for provider in providers:
            srctype = provider.srctype
            assert srctype not in self._providers
            self._providers[srctype] = provider
        if len(self._providers) == 0:
            logger.debug(f"No provider for volume {name}")

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other: 'Volume'):
        return (
            isinstance(other, Volume)
            and self.name == other.name
            and self.space == other.space
            and self.variant == other.variant
            and self._providers == other._providers
        )

    @property
    def name(self):
        """Allows derived classes to implement a lazy name specification."""
        return self._name

    @property
    def providers(self):
        def concat(url: Union[str, Dict[str, str]], concat: str):
            if isinstance(url, str):
                return url + concat
            return {key: url[key] + concat for key in url}
        return {
            srctype: concat(prov._url, f" {prov.label}" if hasattr(prov, "label") else "")
            for srctype, prov in self._providers.items()
        }

    @lru_cache(2)
    def get_boundingbox(self, clip: bool = True, background: float = 0.0, **fetch_kwargs) -> "boundingbox.BoundingBox":
        """
        Obtain the bounding box in physical coordinates of this volume.

        Parameters
        ----------
        clip : bool, default: True
            Whether to clip the background of the volume.
        background : float, default: 0.0
            The background value to clip.
            Note
            ----
            To use it, clip must be True.
        fetch_kwargs:
            key word arguments that are used for fetching volumes,
            such as voi or resolution_mm. Currently, only possible for
            Neuroglancer volumes except for `format`.

        Raises
        ------
        RuntimeError
            If the volume provider does not have a bounding box calculator.
        """
        if self._boundingbox is not None and len(fetch_kwargs) == 0:
            return self._boundingbox

        if not self.provides_image:
            raise NotImplementedError("Bounding box calculation of meshes is not implemented yet.")

        if clip:  # clipping requires fetching the image
            img = self.fetch(**fetch_kwargs)
            assert isinstance(img, Nifti1Image)
            return boundingbox.from_array(
                array=np.asanyarray(img.dataobj),
                background=background,
            ).transform(img.affine, space=self.space)

        # if clipping is not required, providers might have methods of creating
        # bounding boxes without fetching the image
        fmt = fetch_kwargs.get("format")
        if (fmt is not None) and (fmt not in self.formats):
            raise ValueError(
                f"Requested format {fmt} is not available as provider of "
                "this volume. See `volume.formats` for possible options."
            )
        providers = [self._providers[fmt]] if fmt else self._providers.values()
        for provider in providers:
            try:
                assert clip is False
                bbox = provider.get_boundingbox(
                    background=background, **fetch_kwargs
                )
                if bbox.space is None:  # provider do not know the space!
                    bbox._space_cached = self.space
                    bbox.minpoint._space_cached = self.space
                    bbox.maxpoint._space_cached = self.space
            except NotImplementedError:
                continue
            return bbox
        raise RuntimeError(f"No bounding box specified by any volume provider of {str(self)}")

    @property
    def formats(self) -> Set[str]:
        return {fmt for fmt in self._providers}

    @property
    def provides_mesh(self):
        return any(f in self.MESH_FORMATS for f in self.formats)

    @property
    def provides_image(self):
        return any(f in self.IMAGE_FORMATS for f in self.formats)

    @property
    def fragments(self) -> Dict[str, List[str]]:
        result = {}
        for srctype, p in self._providers.items():
            t = 'mesh' if srctype in self.MESH_FORMATS else 'image'
            for fragment_name in p.fragments:
                if t in result:
                    result[t].append(fragment_name)
                else:
                    result[t] = [fragment_name]
        return result

    @property
    def space(self):
        for key in ["@id", "name"]:
            if key in self._space_spec:
                return _space.Space.get_instance(self._space_spec[key])
        return _space.Space(None, "Unspecified space", species=_space.Species.UNSPECIFIED_SPECIES)

    @property
    def species(self):
        s = self.space
        return None if s is None else s.species

    def __str__(self):
        return (
            f"{self.__class__.__name__} {f'{self.name}' if self.name else ''}"
            f"{f' in space {self.space.name}' if self.space else ''}"
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(space_spec={self._space_spec}, "
            f"name='{self.name}', providers={self._providers})>"
        )

    def evaluate_points(
        self,
        points: Union['point.Point', 'pointcloud.PointCloud'],
        outside_value: Union[int, float] = 0,
        **fetch_kwargs
    ) -> np.ndarray:
        """
        Evaluate the image at the positions of the given points.

        Note
        ----
        Uses nearest neighbor interpolation. Other interpolation schemes are not
        yet implemented.

        Note
        ----
        If points are not on the same space as the map, they will be warped to
        the space of the volume.

        Parameters
        ----------
        points: PointCloud
        outside_value: int, float. Default: 0
        fetch_kwargs: dict
            Any additional arguments are passed to the `fetch()` call for
            retrieving the image data.

        Returns
        -------
        values: numpy.ndarray
            The values of the volume at the voxels points correspond to.

        Raises
        ------
        SpaceWarpingFailedError
            If warping of the points fails.
        """
        if not self.provides_image:
            raise NotImplementedError("Filtering of points by pure mesh volumes not yet implemented.")

        # make sure the points are in the same physical space as this volume
        as_pointcloud = pointcloud.from_points([points]) if isinstance(points, point.Point) else points
        warped = as_pointcloud.warp(self.space)
        assert warped is not None, SpaceWarpingFailedError

        # get the voxel array of this volume
        img = self.fetch(format='image', **fetch_kwargs)
        arr = np.asanyarray(img.dataobj)

        # transform the points to the voxel space of the volume for extracting values
        phys2vox = np.linalg.inv(img.affine)
        voxels = warped.transform(phys2vox, space=None)
        XYZ = voxels.coordinates.astype('int')

        # temporarily set all outside voxels to (0,0,0) so that the index access doesn't fail
        inside = np.all((XYZ < arr.shape) & (XYZ > 0), axis=1)
        XYZ[~inside, :] = 0

        # read out the values
        X, Y, Z = XYZ.T
        values = arr[X, Y, Z]

        # fix the outside voxel values, which might have an inconsistent value now
        values[~inside] = outside_value

        return values

    def _points_inside(
        self,
        points: Union['point.Point', 'pointcloud.PointCloud'],
        keep_labels: bool = True,
        outside_value: Union[int, float] = 0,
        **fetch_kwargs
    ) -> 'pointcloud.PointCloud':
        """
        Reduce a pointcloud to the points which fall inside nonzero pixels of
        this map.


        Parameters
        ----------
        points: PointCloud
        keep_labels: bool
            If False, the returned PointCloud will be labeled with their indices
            in the original PointCloud.
        fetch_kwargs: dict
            Any additional arguments are passed to the `fetch()` call for
            retrieving the image data.

        Returns
        -------
        PointCloud
            A new PointCloud containing only the points inside the volume.
            Labels reflect the indices of the original points if `keep_labels`
            is False.
        """
        ptset = pointcloud.from_points([points]) if isinstance(points, point.Point) else points
        values = self.evaluate_points(ptset, outside_value=outside_value, **fetch_kwargs)
        inside = list(np.where(values != outside_value)[0])
        return pointcloud.from_points(
            [ptset[i] for i in inside],
            newlabels=None if keep_labels else inside
        )

    def intersection(self, other: structure.BrainStructure, **fetch_kwargs) -> structure.BrainStructure:
        """
        Compute the intersection of a location with this volume. This will
        fetch actual image data. Any additional arguments are passed to fetch.
        """
        if isinstance(other, (pointcloud.PointCloud, point.Point)):
            try:
                points_inside = self._points_inside(other, keep_labels=False, **fetch_kwargs)
            except EmptyPointCloudError:
                return None  # BrainStructure.intersects checks for not None
            if isinstance(other, point.Point):  # preserve the type
                return points_inside[0]
            return points_inside
        elif isinstance(other, boundingbox.BoundingBox):
            return self.get_boundingbox(clip=True, background=0.0, **fetch_kwargs).intersection(other)
        elif isinstance(other, Volume):
            if self.space != other.space:
                raise NotImplementedError("Cannot intersect volumes from different spaces. Try comparing their bounding boxes.")
            format = fetch_kwargs.pop('format', 'image')
            v1 = self.fetch(format=format, **fetch_kwargs)
            v2 = other.fetch(format=format, **fetch_kwargs)
            arr1 = np.asanyarray(v1.dataobj)
            arr2 = np.asanyarray(resample_img_to_img(v2, v1).dataobj)
            pointwise_min = np.minimum(arr1, arr2)
            if np.any(pointwise_min):
                return from_array(
                    data=pointwise_min,
                    affine=v1.affine,
                    space=self.space,
                    name=f"Intersection between {self} and {other} computed as their pointwise minimum"
                )
            else:
                return None
        else:  # other BrainStructures should have intersection with locations implemented.
            try:
                return other.intersection(self)
            except NoMapAvailableError:
                return None

    def fetch(self, format: str = None, **kwargs):
        """
        Fetch a volumetric or surface representation from one of the providers.

        Parameters
        ----------
        format: str, default=None
            Requested format. If `None`, the first supported format matching in
            `self.formats` is tried, starting with volumetric formats.
            It can be explicitly specified as:
                - 'surface' or 'mesh' to fetch a surface format
                - 'volumetric' or 'voxel' to fetch a volumetric format
                - supported format types, see SUPPORTED_FORMATS. This includes
                'nii', 'zip/nii', 'neuroglancer/precomputed', 'gii-mesh',
                'neuroglancer/precompmesh', 'gii-label'

        Returns
        -------
        An image (Nifti1Image) or mesh (Dict['verts': ndarray, 'faces': ndarray, 'labels': ndarray])
        """
        kwargs_serialized = json.dumps({k: hash(v) for k, v in kwargs.items()}, sort_keys=True)

        if "resolution_mm" in kwargs and format is None:
            if 'neuroglancer/precomputed' not in self.formats:
                raise ValueError("'resolution_mm' is only available for volumes with 'neuroglancer/precomputed' formats.")
            format = 'neuroglancer/precomputed'

        if format is None:
            # preserve fetch order in SUPPORTED_FORMATS
            possible_formats = [f for f in self.SUPPORTED_FORMATS if f in self.formats]
        elif format in self._FORMAT_LOOKUP:  # allow use of aliases
            possible_formats = [f for f in self._FORMAT_LOOKUP[format] if f in self.formats]
        elif format in self.SUPPORTED_FORMATS:
            possible_formats = [format] if format in self.formats else []
        else:
            possible_formats = []
        if len(possible_formats) == 0:
            raise ValueError(
                f"Invalid format requested: {format}. Possible values for this "
                f"volume are: {self.formats}"
            )

        # ensure the voi is inside the template
        voi = kwargs.get("voi", None)
        if voi is not None and voi.space is not None:
            assert isinstance(voi, boundingbox.BoundingBox)
            tmplt_bbox = voi.space.get_template().get_boundingbox(clip=False)
            intersection_bbox = voi.intersection(tmplt_bbox)
            if intersection_bbox is None:
                raise RuntimeError(f"voi provided ({voi}) lies out side the voxel space of the {voi.space.name} template.")
            if intersection_bbox != voi:
                logger.info(
                    f"Since provided voi lies outside the template ({voi.space}) it is clipped as: {intersection_bbox}"
                )
                kwargs["voi"] = intersection_bbox

        result = None
        # try each possible format
        for fmt in possible_formats:
            fetch_hash = hash((hash(self), hash(fmt), hash(kwargs_serialized)))
            # cached
            if fetch_hash in self._FETCH_CACHE:
                break
            # Repeat in case of too many requests only
            fwd_args = {k: v for k, v in kwargs.items() if k != "format"}
            for try_count in range(6):
                try:
                    if fmt in ["gii-label", "freesurfer-annot", "zip/freesurfer-annot"]:
                        tpl = self.space.get_template(variant=kwargs.get('variant'))
                        mesh = tpl.fetch(**kwargs)
                        labels = self._providers[fmt].fetch(**fwd_args)
                        result = dict(**mesh, **labels)
                    else:
                        result = self._providers[fmt].fetch(**fwd_args)
                except requests.SiibraHttpRequestError as e:
                    if e.status_code == 429:  # too many requests
                        sleep(0.1)
                        logger.error(f"Cannot access {self._providers[fmt]}", exc_info=(try_count == 5))
                        continue
                    else:
                        break
                except Exception as e:
                    logger.info(e, exc_info=1)
                    break
                else:
                    break
            # update the cache if fetch is successful
            if result is not None:
                self._FETCH_CACHE[fetch_hash] = result
                while len(self._FETCH_CACHE) >= self._FETCH_CACHE_MAX_ENTRIES:
                    # remove oldest entry
                    self._FETCH_CACHE.pop(next(iter(self._FETCH_CACHE)))
                break
        else:
            # unsuccessful: do not poison the cache if none fetched
            logger.error(f"Could not fetch any formats from {possible_formats}.")
            return None

        return self._FETCH_CACHE[fetch_hash]

    def fetch_connected_components(self, **fetch_kwargs):
        """
        Provide an generator over masks of connected components in the volume
        """
        img = self.fetch(**fetch_kwargs)
        assert isinstance(img, Nifti1Image), NotImplementedError(
            f"Connected components for type {type(img)} is not yet implemented."
        )
        for label, component in connected_components(np.asanyarray(img.dataobj)):
            yield (
                label,
                Nifti1Image(component, img.affine)
            )

    def compute_spatial_props(self, split_components: bool = True, **fetch_kwargs) -> List[ComponentSpatialProperties]:
        """
        Find the center of this volume in its (non-zero) voxel space and and its
        volume.

        Parameters
        ----------
        split_components: bool, default: True
            If True, finds the spatial properties for each connected component
            found by skimage.measure.label.
        """
        assert self.provides_image, NotImplementedError("Spatial properties can currently on be calculated for images.")
        img = self.fetch(format=fetch_kwargs.pop("format", "image"), **fetch_kwargs)
        return ComponentSpatialProperties.compute_from_image(
            img=img,
            space=self.space,
            split_components=split_components
        )

    def draw_samples(self, N: int, sample_size: int = 100, e: float = 1, sigma_mm=None, invert=False, **kwargs):
        """
        Draw samples from the volume, by interpreting its values as an
        unnormalized empirical probability distributions.
        Any keyword arguments are passed over to fetch()
        """
        if not self.provides_image:
            raise NotImplementedError(
                "Drawing samples is so far only implemented for image-type volumes, "
                f"not {self.__class__.__name__}."
            )
        img = self.fetch(**kwargs)
        array = np.asanyarray(img.dataobj)
        samples = []
        P = (array - array.min()) / (array.max() - array.min())
        if invert:
            P = 1 - P
        P = P**e
        while True:
            pts = (np.random.rand(sample_size, 3) * max(P.shape))
            inside = np.all(pts < P.shape, axis=1)
            Y, X, Z = np.split(pts[inside, :].astype('int'), 3, axis=1)
            T = np.random.rand(1)
            choice = np.where(P[Y, X, Z] >= T)[0]
            samples.extend(list(pts[inside, :][choice, :]))
            if len(samples) > N:
                break
        voxels = pointcloud.PointCloud(
            np.random.permutation(samples)[:N, :],
            space=None
        )
        result = voxels.transform(img.affine, space='mni152')
        result.sigma_mm = [sigma_mm for _ in result]
        return result

    def find_peaks(self, mindist=5, sigma_mm=0, **kwargs):
        """
        Find local peaks in the volume.
        Additional keyword arguments are passed over to fetch()
        """
        if not self.provides_image:
            raise NotImplementedError(
                "Finding peaks is so far only implemented for image-type volumes, "
                f"not {self.__class__.__name__}."
            )
        img = self.fetch(**kwargs)
        array = np.asanyarray(img.dataobj)
        voxels = skimage_feature.peak_local_max(array, min_distance=mindist)
        points = pointcloud.PointCloud(voxels, space=None, labels=list(range(len(voxels)))).transform(img.affine, space=self.space)
        points.sigma_mm = [sigma_mm for _ in points]
        return points


class FilteredVolume(Volume):

    def __init__(
        self,
        parent_volume: Volume,
        label: int = None,
        fragment: str = None,
        threshold: float = None,
    ):
        """
        A prescribed Volume to fetch specified label and fragment.
        If threshold is defined, a mask of the values above the threshold.

        Parameters
        ----------
        parent_volume : Volume
        label : int, default: None
            Get the mask of value equal to label.
        fragment : str, default None
            If a volume is fragmented, get a specified one.
        threshold : float, default None
            Provide a float value to threshold the image.
        """
        name = parent_volume.name
        if label:
            name += f" - label: {label}"
        if fragment:
            name += f" - fragment: {fragment}"
        if threshold:
            name += f" - threshold: {threshold}"
        Volume.__init__(
            self,
            space_spec=parent_volume._space_spec,
            providers=list(parent_volume._providers.values()),
            name=name
        )
        self.fragment = fragment
        self.label = label
        self.threshold = threshold

    def fetch(
        self,
        format: str = None,
        **kwargs
    ):
        if "fragment" in kwargs:
            assert kwargs.get("fragment") == self.fragment, f"This is a filtered volume that can only fetch fragment '{self.fragment}'."
        else:
            kwargs["fragment"] = self.fragment
        if "label" in kwargs:
            assert kwargs.get("label") == self.label, f"This is a filtered volume that can only fetch label '{self.label}' only."
        else:
            kwargs["label"] = self.label

        result = super().fetch(format=format, **kwargs)

        if self.threshold is not None:
            assert self.label is None
            if not isinstance(result, Nifti1Image):
                raise NotImplementedError("Cannot threshold meshes.")
            imgdata = np.asanyarray(result.dataobj)
            return Nifti1Image(
                dataobj=(imgdata > self.threshold).astype("uint8"),
                affine=result.affine,
                dtype="uint8"
            )

        return result

    def get_boundingbox(
        self,
        clip: bool = True,
        background: float = 0.0,
        **fetch_kwargs
    ) -> "boundingbox.BoundingBox":
        # NOTE: since some providers enable different simpllified ways to create a
        # bounding box without fetching the image, the correct kwargs must be
        # forwarded since FilteredVolumes enforce their specs to be fetched.
        return super().get_boundingbox(
            clip=clip,
            background=background,
            **fetch_kwargs
        )


class ReducedVolume(Volume):
    def __init__(
        self,
        source_volumes: List[Volume],
        new_labels: List[int] = None,
    ):
        """
        A prescribed Volume to fetch a list of source volumes in the same space,
        with optionally new labels, and resampling them to the template of this
        space to be merged into one image.

        Parameters
        ----------
        source_volumes: List[Volume]
        new_labels: List[int], default: None
        """
        assert len({v.space for v in source_volumes}) == 1, "Cannot merge volumes from different spaces."
        if new_labels:
            assert len(source_volumes) == len(new_labels), "Need to supply as many labels as volumes."
        Volume.__init__(
            self,
            space_spec=source_volumes[0]._space_spec,
            providers=[],
            name=f"Volumes resampled to {source_volumes[0].space} template and merged: {','.join([v.name for v in source_volumes])}"
        )
        self.source_volumes = source_volumes
        self.new_labels = new_labels

    @lru_cache(2)
    def fetch(self, format: str = None, **kwargs):
        # determine dtype
        if self.new_labels is not None:
            logger.info(f"Relabling regions with labels: {self.new_labels}")
            dtype = 'int32'
        elif {type(v) for v in self.source_volumes} == {FilteredVolume}:
            dtype = 'uint8'
        else:
            dtype = self.source_volumes[0].fetch().dataobj.dtype

        # determine base image
        template_img = self.space.get_template().fetch(**kwargs)
        template_res = min(template_img.header.get_zooms())
        if "neuroglancer/precomputed" in self.formats:
            kwargs["resolution_mm"] = kwargs.get("resolution_mm", template_res)
        merged_array = np.zeros(template_img.shape, dtype=dtype)
        for i, vol in siibra_tqdm(
            enumerate(self.source_volumes),
            unit=" volume",
            desc=f"Resampling volumes to {self.space.name} and merging",
            total=len(self.source_volumes),
            disable=len(self.source_volumes) < 3,
            leave=False,
        ):
            img = vol.fetch(format=format, **kwargs)
            if img is None:
                continue
            resampled_arr = np.asanyarray(
                resample_img_to_img(img, template_img).dataobj
            )
            nonzero_voxels = resampled_arr > 0
            if self.new_labels:
                merged_array[nonzero_voxels] = self.new_labels[i]
            else:
                merged_array[nonzero_voxels] = resampled_arr[nonzero_voxels]

        result = Nifti1Image(merged_array, affine=template_img.affine, dtype=dtype)
        if 'neuroglancer/precomputed' in self.formats:
            result.set_qform(result.affine)
        return result

    def get_boundingbox(
        self,
        clip: bool = True,
        background: float = 0.0,
        **fetch_kwargs
    ) -> boundingbox.BoundingBox:
        if clip:
            with QUIET:
                return super().get_boundingbox(clip=clip, background=background, **fetch_kwargs)
        else:
            return boundingbox.BoundingBox.union(
                *[
                    v.get_boundingbox(clip=clip, background=background, **fetch_kwargs)
                    for v in self.source_volumes
                ]
            )

    @property
    def formats(self) -> Set[str]:
        return {fmt for v in self.source_volumes for fmt in v._providers}


class Subvolume(Volume):
    """
    Wrapper class for exposing a z level of a 4D volume to be used like a 3D volume.
    """

    def __init__(self, parent_volume: Volume, z: int):
        Volume.__init__(
            self,
            space_spec=parent_volume._space_spec,
            providers=[
                _providers.provider.SubvolumeProvider(p, z=z)
                for p in parent_volume._providers.values()
            ],
            name=parent_volume.name + f" - z: {z}"
        )


def from_file(filename: str, space: str, name: str) -> Volume:
    """Builds a nifti volume from a filename."""
    spaceobj = get_registry("Space").get(space)
    return Volume(
        space_spec={"@id": spaceobj.id},
        providers=[_providers.NiftiProvider(filename)],
        name=filename if name is None else name,
    )


def from_nifti(nifti: Nifti1Image, space: str, name: str) -> Volume:
    """Builds a nifti volume from a Nifti image."""
    spaceobj = get_registry("Space").get(space)
    return Volume(
        space_spec={"@id": spaceobj.id},
        providers=[_providers.NiftiProvider((np.asanyarray(nifti.dataobj), nifti.affine))],
        name=name
    )


def from_array(
    data: np.ndarray,
    affine: np.ndarray,
    space: Union[str, Dict[str, str]],
    name: str
) -> Volume:
    """Builds a siibra volume from an array and an affine matrix."""
    if len(name) == 0:
        raise ValueError("Please provide a non-empty string for `name`")
    spacespec = next(iter(space.values())) if isinstance(space, dict) else space
    spaceobj = get_registry("Space").get(spacespec)
    return Volume(
        space_spec={"@id": spaceobj.id},
        providers=[_providers.NiftiProvider((data, affine))],
        name=name,
    )


def from_pointcloud(
    points: pointcloud.PointCloud,
    label: int = None,
    target: Volume = None,
    normalize=True,
    **kwargs
) -> Volume:
    """
    Get the kernel density estimate as a volume from the points using their
    average uncertainty on target volume.

    Parameters
    ----------
    points: pointcloud.PointCloud
    label: int, default: None
        If None, finds the KDE for all points. Otherwise, selects the points
        labelled with this integer value.
    target: Volume, default: None
        If None, the template of the space points are defined on will be used.
    normalize: bool, default: True

    Raises
    ------
    RuntimeError
        If no points with labels found
    """
    if target is None:
        target = points.space.get_template()
    targetimg = target.fetch(**kwargs)
    voxels = points.transform(np.linalg.inv(targetimg.affine), space=None)

    if (label is None) or (points.labels is None):
        selection = [True for _ in points]
    else:
        assert label in points.labels, f"No points with the label {label} in the set: {set(points.labels)}"
        selection = points.labels == label

    voxelcount_img = np.zeros_like(targetimg.get_fdata())
    unique_coords, counts = np.unique(
        np.array(voxels.as_list(), dtype='int')[selection, :],
        axis=0,
        return_counts=True
    )
    voxelcount_img[tuple(unique_coords.T)] = counts

    # TODO: consider how to handle pointclouds with varied sigma_mm
    sigmas = np.array(points.sigma_mm)[selection]
    bandwidth = np.mean(sigmas)
    if len(np.unique(sigmas)) > 1:
        logger.warning(f"KDE of pointcloud uses average bandwidth {bandwidth} instead of the points' individual sigmas.")

    filtered_arr = filters.gaussian(voxelcount_img, bandwidth)
    if normalize:
        filtered_arr /= filtered_arr.sum()

    return from_array(
        data=filtered_arr,
        affine=targetimg.affine,
        space=target.space,
        name=f'KDE map of {points}{f"labelled {label}" if label else ""}'
    )


def merge(volumes: List[Volume], labels: List[int] = [], **fetch_kwargs) -> Volume:
    """
    Merge a list of nifti volumes in the same space into a single volume.

    Note
    ----
    In case of voxel conflicts, the volumes will be override the previous values
    in the given order.

    Parameters
    ----------
    volumes : List[Volume]
    labels : List[int], optional
        Supply new labels to replace existing values per volume.

    Returns
    -------
    Volume
    """
    if len(volumes) == 1:
        logger.debug("Only one volume supplied returning as is (kwargs are ignored).")
        return volumes[0]

    assert len(volumes) > 1, "Need to supply at least two volumes to merge."
    if labels:
        assert len(volumes) == len(labels), "Need to supply as many labels as volumes."

    space = volumes[0].space
    assert all(v.space == space for v in volumes), "Cannot merge volumes from different spaces."

    if len(labels) > 0:
        dtype = 'int32'
    elif FilteredVolume in {type(v) for v in volumes}:
        dtype = 'uint8'
    else:
        dtype = volumes[0].fetch().dataobj.dtype
    template_img = space.get_template().fetch(**fetch_kwargs)
    merged_array = np.zeros(template_img.shape, dtype=dtype)

    for i, vol in siibra_tqdm(
        enumerate(volumes),
        unit=" volume",
        desc=f"Resampling volumes to {space.name} and merging",
        total=len(volumes),
        disable=len(volumes) < 3
    ):
        img = vol.fetch(**fetch_kwargs)
        resampled_arr = np.asanyarray(
            resample_img_to_img(img, template_img).dataobj
        )
        nonzero_voxels = resampled_arr > 0
        if labels:
            merged_array[nonzero_voxels] = labels[i]
        else:
            merged_array[nonzero_voxels] = resampled_arr[nonzero_voxels]

    return from_array(
        data=merged_array,
        affine=template_img.affine,
        space=space,
        name=f"Resampled and merged volumes: {','.join([v.name for v in volumes])}"
    )
