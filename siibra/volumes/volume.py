# Copyright 2018-2021
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

from .providers import provider as _provider

from .. import logger
from ..retrieval import requests
from ..core import space as _space, structure
from ..locations import location, point, pointset, boundingbox
from ..commons import resample_array_to_array
from ..exceptions import NoMapAvailableError, SpaceWarpingFailedError

from nibabel import Nifti1Image
import numpy as np
from typing import List, Dict, Union, Set, TYPE_CHECKING
from time import sleep
import json
from skimage import filters, feature as skimage_feature

if TYPE_CHECKING:
    from ..retrieval.datasets import EbrainsDataset
    TypeDataset = EbrainsDataset


class Volume(structure.BrainStructure, location.Location):
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
        "gii-label"
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
        providers: List['_provider.VolumeProvider'],
        name: str = "",
        variant: str = None,
        datasets: List['TypeDataset'] = [],
    ):
        self._name_cached = name  # see lazy implementation below
        self._space_spec = space_spec
        self.variant = variant
        self._providers: Dict[str, _provider.VolumeProvider] = {}
        self.datasets = datasets
        for provider in providers:
            srctype = provider.srctype
            assert srctype not in self._providers
            self._providers[srctype] = provider
        if len(self._providers) == 0:
            logger.debug(f"No provider for volume {self}")

    def __hash__(self):
        """Enrich the default hash with the name of the volume."""
        return hash(self.name) ^ super().__hash__()

    @property
    def name(self):
        """
        Allows derived classes to implement a lazy name specification.
        """
        return self._name_cached

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

    @property
    def boundingbox(self):
        for provider in self._providers.values():
            try:
                bbox = provider.boundingbox
                if bbox.space is None:  # provider does usually not know the space!
                    bbox._space_cached = self.space
                    bbox.minpoint._space_cached = self.space
                    bbox.maxpoint._space_cached = self.space
            except NotImplementedError as e:
                print(str(e))
                continue
            return bbox
        raise RuntimeError(f"No bounding box specified by any volume provider of {str(self)}")

    @property
    def formats(self) -> Set[str]:
        result = set()
        for fmt in self._providers:
            result.add(fmt)
            result.add('mesh' if fmt in self.MESH_FORMATS else 'image')
        return result

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
        if self.space is None:
            return f"{self.__class__.__name__} '{self.name}'"
        else:
            return f"{self.__class__.__name__} '{self.name}' in space '{self.space.name}'"

    def __repr__(self):
        return self.__str__()

    def points_inside(self, points: pointset.PointSet, **kwargs) -> List[int]:
        """
        Reduce a pointset to the points which fall
        inside nonzero pixels of this volume.
        The indices of the original points are stored as point labels in the result.
        Any additional arguments are passed to the fetch() call
        for retrieving the image data.
        """
        if not self.provides_image:
            raise NotImplementedError("Filtering of points by pure mesh volumes not yet implemented.")
        img = self.fetch(format='image', **kwargs)
        arr = img.get_fdata()
        warped = points.warp(self.space)
        assert warped is not None
        phys2vox = np.linalg.inv(img.affine)
        voxels = warped.transform(phys2vox, space=None)
        XYZ = voxels.homogeneous.astype('int')[:, :3]
        invalid = np.where(
            np.all(XYZ >= arr.shape, axis=1)
            | np.all(XYZ < 0, axis=1)
        )[0]
        XYZ[invalid] = 0  # set all out-of-bounds vertices to (0, 0, 0)
        arr[0, 0, 0] = 0  # ensure the lower left voxel is not foreground
        inside = np.where(arr[tuple(zip(*XYZ))] != 0)[0]
        return pointset.PointSet(
            points.homogeneous[inside, :3],
            space=points.space,
            labels=inside
        )

    def intersection(self, other: structure.BrainStructure, **kwargs) -> structure.BrainStructure:
        """
        Compute the intersection of a location with this volume.
        This will fetch actual image data.
        Any additional arguments are passed to fetch.
        TODO write a test for the volume-volume and volume-region intersection
        """
        if isinstance(other, (pointset.PointSet, point.Point)):
            result = self.points_inside(other, **kwargs)
            if len(result) == 0:
                return pointset.PointSet([], space=other.space)
            elif len(result) == 1:
                return result[0]
            else:
                return result
        elif isinstance(other, boundingbox.BoundingBox):
            return self.boundingbox.intersection(other)
        elif isinstance(other, Volume):
            format = kwargs.pop('format', 'image')
            v1 = self.fetch(format=format, **kwargs)
            v2 = other.fetch(format=format, **kwargs)
            arr1 = np.asanyarray(v1.dataobj)
            arr2 = resample_array_to_array(np.asanyarray(v2.dataobj), v2.affine, arr1, v1.affine)
            pointwise_min = np.minimum(arr1, arr2)
            if np.any(pointwise_min):
                return from_array(
                    pointwise_min,
                    v1.affine, self.space,
                    name=f"Intersection between {self} and {other} computed as their pointwise minimum"
                )
            else:
                return None
        else:  # other BrainStructures should have intersection with locations implemented.
            try:
                return other.intersection(self)
            except NoMapAvailableError:
                return None

    def transform(self, affine: np.ndarray, space=None):
        """ only modifies the affine matrix and space. """
        return Volume(
            spacespec=space,
            providers=[p.transform(affine, space=space) for p in self.providers],
            name=self.name,
            variant=self.variant,
            datasets=self.datasets
        )

    def warp(self, space):
        if self.space.matches(space):
            return self
        else:
            raise SpaceWarpingFailedError('Warping of full volumes is not yet supported.')

    def fetch(
        self,
        format: str = None,
        **kwargs
    ):
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
        An image or mesh
        """
        # check for a cached object
        fetch_hash = hash(
            (self.__hash__(), hash(format), hash(json.dumps(kwargs, sort_keys=True)))
        )
        if fetch_hash in self._FETCH_CACHE:
            return self._FETCH_CACHE[fetch_hash]

        result = None

        # no cached object, fetch now
        if format is None:
            requested_formats = list(self._providers.keys())
        elif format in self._FORMAT_LOOKUP:  # allow use of aliases
            requested_formats = self._FORMAT_LOOKUP[format]
        elif format in self.SUPPORTED_FORMATS:
            requested_formats = [format]
        else:
            raise ValueError(f"Invalid format requested: {format}")

        possible_formats = set(requested_formats) & set(self.formats)
        if len(possible_formats) == 0:
            raise ValueError(f"Invalid format requested: {format}")

        # try the selected format only
        for selected_format in possible_formats:
            fwd_args = {k: v for k, v in kwargs.items() if k != "format"}
            try:
                if selected_format == "gii-label":
                    tpl = self.space.get_template(variant=kwargs.get('variant'))
                    mesh = tpl.fetch(**kwargs)
                    labels = self._providers[selected_format].fetch(**fwd_args)
                    result = dict(**mesh, **labels)
                    break
                else:
                    assert selected_format in self._providers
                    result = self._providers[selected_format].fetch(**fwd_args)
                    # TODO insert a meaningful description including origin and bounding box
                    break
            except requests.SiibraHttpRequestError as e:
                if e.status_code == 429:  # too many requests
                    sleep(0.1)
                logger.error(f"Cannot access {self._providers[selected_format]}")

        while len(self._FETCH_CACHE) >= self._FETCH_CACHE_MAX_ENTRIES:
            # remove oldest entry
            self._FETCH_CACHE.pop(next(iter(self._FETCH_CACHE)))
        self._FETCH_CACHE[fetch_hash] = result
        return result

    def fetch_connected_components(self, **kwargs):
        """
        Provide an iterator over masks of connected components in the volume
        """
        img = self.fetch(**kwargs)
        from skimage import measure
        imgdata = np.asanyarray(img.dataobj).squeeze()
        components = measure.label(imgdata > 0)
        component_labels = np.unique(components)
        assert component_labels[0] == 0
        return (
            (label, Nifti1Image((components == label).astype('uint8'), img.affine))
            for label in component_labels[1:]
        )

    def draw_samples(self, N: int, sample_size: int = 100, e: float = 1, sigma_mm=None, invert=False, **kwargs):
        """
        Draw samples from the volume, by interpreting its values as an
        unnormalized empirical probability distribtution.
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
        voxels = pointset.PointSet(
            np.random.permutation(samples)[:N, :],
            space=None
        )
        result = voxels.transform(img.affine, space='mni152')
        result.sigma_mm = sigma_mm
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
        points = pointset.PointSet(voxels, space=None, labels=list(range(len(voxels)))).transform(img.affine, space=self.space)
        points.sigma_mm = [sigma_mm for _ in points]
        return points


class Subvolume(Volume):
    """
    Wrapper class for exposing a z level of a 4D volume to be used like a 3D volume.
    """

    def __init__(self, parent_volume: Volume, z: int):
        Volume.__init__(
            self,
            space_spec=parent_volume._space_spec,
            providers=[
                _provider.SubvolumeProvider(p, z=z)
                for p in parent_volume._providers.values()
            ]
        )


def from_file(filename: str, space: str, name: str = None):
    """ Builds a nifti volume from a filename. """
    from ..core.concept import AtlasConcept
    from .providers.nifti import NiftiProvider
    spaceobj = AtlasConcept.get_registry("Space").get(space)
    return Volume(
        space_spec={"@id": spaceobj.id},
        providers=[NiftiProvider(filename)],
        name=filename if name is None else name,
    )


def from_array(
    data: np.ndarray,
    affine: np.ndarray,
    space: Union[str, Dict[str, str]],
    name: str = ""
):
    """ Builds a siibra volume from an array and an affine matrix. """
    from ..core.concept import AtlasConcept
    from .providers.nifti import NiftiProvider
    spacespec = next(iter(space.values())) if isinstance(space, dict) else space
    spaceobj = AtlasConcept.get_registry("Space").get(spacespec)
    return Volume(
        space_spec={"@id": spaceobj.id},
        providers=[NiftiProvider((data, affine))],
        name=name,
    )


def from_pointset(
    points: pointset.PointSet,
    label: int,
    target: Volume,
    min_num_points=10,
    normalize=True,
    **kwargs
):
    targetimg = target.fetch(**kwargs)
    voxels = points.transform(np.linalg.inv(targetimg.affine), space=None)
    selection = [_ == label for _ in points.labels]
    if np.count_nonzero(selection) == 0:
        raise RuntimeError(f"No points with label {label} in the set: {', '.join(map(str, points.labels))}")
    X, Y, Z = np.split(
        np.array(voxels.as_list()).astype('int')[selection, :],
        3, axis=1
    )
    if len(X) < min_num_points:
        return None
    cimg = np.zeros_like(targetimg.get_fdata())
    cimg[X, Y, Z] += 1
    if isinstance(points.sigma_mm, (int, float)):
        bandwidth = points.sigma_mm
    elif isinstance(points.sigma_mm, list):
        logger.debug(
            "Computing kernel density estimate from pointset using their average uncertainty."
        )
        bandwidth = np.sum(points.sigma_mm) / len(points)
    else:
        logger.warn("Poinset has no uncertainty, using bandwith=1mm for kernel density estimate.")
        bandwidth = 1
    data = filters.gaussian(cimg, bandwidth)
    if normalize:
        data /= data.sum()
    return from_array(
        data,
        affine=targetimg.affine,
        space=target.space,
        name=f'KDE map of {sum(selection)} points with label={label}'
    )
