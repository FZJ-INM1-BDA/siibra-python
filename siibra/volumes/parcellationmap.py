# Copyright 2018-2023
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
"""Provides spatial representations for parcellations and regions."""

from . import volume as _volume
from .providers import provider
from .. import logger, QUIET, exceptions
from ..commons import (
    MapIndex,
    MapType,
    compare_arrays,
    resample_array_to_array,
    connected_components,
    clear_name,
    create_key,
    create_gaussian_kernel,
    siibra_tqdm,
    Species,
    CompareMapsResult,
    generate_uuid
)
from ..core import concept, space, parcellation, region as _region
from ..locations import location, point, pointset
from ..retrieval import requests

import numpy as np
from typing import Union, Dict, List, TYPE_CHECKING, Iterable, Tuple
from scipy.ndimage import distance_transform_edt
from collections import defaultdict
from nilearn import image
import pandas as pd
from dataclasses import dataclass, asdict

if TYPE_CHECKING:
    from ..core.region import Region


@dataclass
class MapAssignment:
    input_structure: int
    centroid: Union[Tuple[np.ndarray], point.Point]
    volume: int
    fragment: str
    map_value: np.ndarray


@dataclass
class AssignImageResult(CompareMapsResult, MapAssignment):
    pass


class Map(concept.AtlasConcept, configuration_folder="maps"):

    def __init__(
        self,
        identifier: str,
        name: str,
        space_spec: dict,
        parcellation_spec: dict,
        indices: Dict[str, List[Dict]],
        volumes: list = [],
        shortname: str = "",
        description: str = "",
        modality: str = None,
        publications: list = [],
        datasets: list = [],
    ):
        """
        Constructs a new parcellation object.

        Parameters
        ----------
        identifier: str
            Unique identifier of the parcellation
        name: str
            Human-readable name of the parcellation
        space_spec: dict
            Specification of the space (use @id or name fields)
        parcellation_spec: str
            Specification of the parcellation (use @id or name fields)
        indices: dict
            Dictionary of indices for the brain regions.
            Keys are exact region names.
            Per region name, a list of dictionaries with fields "volume" and "label" is expected,
            where "volume" points to the index of the Volume object where this region is mapped,
            and optional "label" is the voxel label for that region.
            For continuous / probability maps, the "label" can be null or omitted.
            For single-volume labelled maps, the "volume" can be null or omitted.
        volumes: list[Volume]
            parcellation volumes
        shortname: str, optional
            Shortform of human-readable name
        description: str, optional
            Textual description of the parcellation
        modality: str, default: None
            Specification of the modality used for creating the parcellation
        publications: list
            List of associated publications, each a dictionary with "doi" and/or "citation" fields
        datasets : list
            datasets associated with this concept
        """
        concept.AtlasConcept.__init__(
            self,
            identifier=identifier,
            name=name,
            species=None,  # inherits species from space
            shortname=shortname,
            description=description,
            publications=publications,
            datasets=datasets,
            modality=modality
        )
        self._space_spec = space_spec
        self._parcellation_spec = parcellation_spec
        if 'prerelease' in self.parcellation.name.lower():
            self.name = f"[PRERELEASE] {self.name}"

        # Since the volumes might include 4D arrays, where the actual
        # volume index points to a z coordinate, we create subvolume
        # indexers from the given volume provider if 'z' is specified.
        self._indices: Dict[str, List[MapIndex]] = {}
        self.volumes: List[_volume.Volume] = []
        remap_volumes = {}
        # TODO: This assumes knowledge of the preconfigruation specs wrt. z.
        # z to subvolume conversion should probably go to the factory.
        for regionname, indexlist in indices.items():
            k = clear_name(regionname)
            self._indices[k] = []
            for index in indexlist:
                vol = index.get('volume', 0)
                assert vol in range(len(volumes))
                z = index.get('z')
                if (vol, z) not in remap_volumes:
                    if z is None:
                        self.volumes.append(volumes[vol])
                    else:
                        self.volumes.append(_volume.Subvolume(volumes[vol], z))
                    remap_volumes[vol, z] = len(self.volumes) - 1
                self._indices[k].append(
                    MapIndex(volume=remap_volumes[vol, z], label=index.get('label'), fragment=index.get('fragment'))
                )

        # make sure the indices are unique - each map/label pair should appear at most once
        all_indices = sum(self._indices.values(), [])
        seen = set()
        duplicates = {x for x in all_indices if x in seen or seen.add(x)}
        if len(duplicates) > 0:
            logger.warning(f"Non unique indices encountered in {self}: {duplicates}")
        self._affine_cached = None

    @property
    def species(self) -> Species:
        # lazy implementation
        if self._species_cached is None:
            self._species_cached = self.space.species
        return self.space._species_cached

    def get_index(self, region: Union[str, "Region"]):
        """
        Returns the unique index corresponding to the specified region.

        Tip
        ----
        Use find_indices() method for a less strict search returning all matches.

        Parameters
        ----------
        region: str or Region

        Returns
        -------
        MapIndex

        Raises
        ------
        NonUniqueIndexError
            If not unique or not defined in this parcellation map.
        """
        matches = self.find_indices(region)
        if len(matches) > 1:
            # if there is an exact match, we still use it. If not, we cannot proceed.
            regionname = region.name if isinstance(region, _region.Region) \
                else region
            for index, matched_name in matches.items():
                if matched_name == regionname:
                    return index
            raise exceptions.NonUniqueIndexError(
                f"The specification '{region}' matches multiple mapped "
                f"structures in {str(self)}: {list(matches.values())}"
            )
        elif len(matches) == 0:
            raise exceptions.NonUniqueIndexError(
                f"The specification '{region}' does not match to any structure mapped in {self}"
            )
        else:
            return next(iter(matches))

    def find_indices(self, region: Union[str, "Region"]):
        """
        Returns the volume/label indices in this map which match the given
        region specification.

        Parameters
        ----------
        region: str or Region

        Returns
        -------
        dict
            - keys: MapIndex
            - values: region name
        """
        if region in self._indices:
            return {
                idx: region
                for idx in self._indices[region]
            }
        regionname = region.name if isinstance(region, _region.Region) else region
        matched_region_names = set(_.name for _ in (self.parcellation.find(regionname)))
        matches = matched_region_names & self._indices.keys()
        if len(matches) == 0:
            logger.warning(f"Region {regionname} not defined in {self}")
        return {
            idx: regionname
            for regionname in matches
            for idx in self._indices[regionname]
        }

    def get_region(self, label: int = None, volume: int = 0, index: MapIndex = None):
        """
        Returns the region mapped by the given index, if any.

        Tip
        ----
        Use get_index() or find_indices() methods to obtain the MapIndex.

        Parameters
        ----------
        label: int, default: None
        volume: int, default: 0
        index: MapIndex, default: None

        Returns
        -------
        Region
            A region object defined in the parcellation map.
        """
        if isinstance(label, MapIndex) and index is None:
            raise TypeError("Specify MapIndex with 'index' keyword.")
        if index is None:
            index = MapIndex(volume, label)
        matches = [
            regionname
            for regionname, indexlist in self._indices.items()
            if index in indexlist
        ]
        if len(matches) == 0:
            logger.warning(f"Index {index} not defined in {self}")
            return None
        elif len(matches) == 1:
            return self.parcellation.get_region(matches[0])
        else:
            # this should not happen, already tested in constructor
            raise RuntimeError(f"Index {index} is not unique in {self}")

    @property
    def space(self):
        for key in ["@id", "name"]:
            if key in self._space_spec:
                return space.Space.get_instance(self._space_spec[key])
        return space.Space(None, "Unspecified space", species=Species.UNSPECIFIED_SPECIES)

    @property
    def parcellation(self):
        for key in ["@id", "name"]:
            if key in self._parcellation_spec:
                return parcellation.Parcellation.get_instance(self._parcellation_spec[key])
        logger.warning(
            f"Cannot determine parcellation of {self.__class__.__name__} "
            f"{self.name} from {self._parcellation_spec}"
        )
        return None

    @property
    def labels(self):
        """
        The set of all label indices defined in this map, including "None" if
        not defined for one or more regions.
        """
        return {d.label for v in self._indices.values() for d in v}

    @property
    def maptype(self) -> MapType:
        if all(isinstance(_, int) for _ in self.labels):
            return MapType.LABELLED
        elif self.labels == {None}:
            return MapType.STATISTICAL
        else:
            raise RuntimeError(
                f"Inconsistent label indices encountered in {self}"
            )

    def __len__(self):
        return len(self.volumes)

    @property
    def regions(self):
        return list(self._indices)

    def fetch(
        self,
        region_or_index: Union[str, "Region", MapIndex] = None,
        *,
        index: MapIndex = None,
        region: Union[str, "Region"] = None,
        **kwargs
    ):
        """
        Fetches one particular volume of this parcellation map.

        If there's only one volume, this is the default, otherwise further
        specification is requested:
        - the volume index,
        - the MapIndex (which results in a regional map being returned)

        You might also consider fetch_iter() to iterate the volumes, or
        compress() to produce a single-volume parcellation map.

        Parameters
        ----------
        region_or_index: str, Region, MapIndex
            Lazy match the specification.
        index: MapIndex
            Explicit specification of the map index, typically resulting
            in a regional map (mask or statistical map) to be returned.
            Note that supplying 'region' will result in retrieving the map index of that region
            automatically.
        region: str, Region
            Specification of a region name, resulting in a regional map
            (mask or statistical map) to be returned.
        **kwargs
            - resolution_mm: resolution in millimeters
            - format: the format of the volume, like "mesh" or "nii"
            - voi: a BoundingBox of interest


            Note
            ----
            Not all keyword arguments are supported for volume formats. Format
            is restricted by available formats (check formats property).

        Returns
        -------
        An image or mesh
        """
        try:
            length = len([arg for arg in [region_or_index, region, index] if arg is not None])
            assert length == 1
        except AssertionError:
            if length > 1:
                raise exceptions.ExcessiveArgumentException("One and only one of region_or_index, region, index can be defined for fetch")
            # user can provide no arguments, which assumes one and only one volume present

        if isinstance(region_or_index, MapIndex):
            index = region_or_index

        if isinstance(region_or_index, (str, _region.Region)):
            region = region_or_index

        mapindex = None
        if region is not None:
            assert isinstance(region, (str, _region.Region))
            mapindex = self.get_index(region)
        if index is not None:
            assert isinstance(index, MapIndex)
            mapindex = index
        if mapindex is None:
            if len(self) == 1:
                mapindex = MapIndex(volume=0, label=None)
            elif len(self) > 1:
                logger.info(
                    "Map provides multiple volumes and no specification is"
                    " provided. Resampling all volumes to the space."
                )
                labels = list(range(len(self.volumes)))
                merged_volume = _volume.merge(self.volumes, labels, **kwargs)
                return merged_volume.fetch()
            else:
                raise exceptions.NoVolumeFound("Map provides no volumes.")

        if "resolution_mm" in kwargs and kwargs.get("format") is None:
            kwargs["format"] = 'neuroglancer/precomputed'

        kwargs_fragment = kwargs.pop("fragment", None)
        if kwargs_fragment is not None:
            if (mapindex.fragment is not None) and (kwargs_fragment != mapindex.fragment):
                raise exceptions.ConflictingArgumentException(
                    "Conflicting specifications for fetching volume fragment: "
                    f"{mapindex.fragment} / {kwargs_fragment}"
                )
            mapindex.fragment = kwargs_fragment

        if mapindex.volume is None:
            mapindex.volume = 0
        if mapindex.volume >= len(self.volumes):
            raise IndexError(
                f"{self} provides {len(self)} mapped volumes, but #{mapindex.volume} was requested."
            )
        try:
            result = self.volumes[mapindex.volume or 0].fetch(
                fragment=mapindex.fragment, label=mapindex.label, **kwargs
            )
        except requests.SiibraHttpRequestError as e:
            print(str(e))

        if result is None:
            raise RuntimeError(f"Error fetching {mapindex} from {self} as {kwargs.get('format', f'{self.formats}')}.")

        return result

    def fetch_iter(self, **kwargs):
        """
        Returns an iterator to fetch all mapped volumes sequentially.

        All arguments are passed on to function Map.fetch(). By default, it
        will go through all fragments as well.
        """
        fragments = {kwargs.pop('fragment', None)} or self.fragments or {None}
        return (
            self.fetch(
                index=MapIndex(volume=i, label=None, fragment=frag), **kwargs
            )
            for frag in fragments
            for i in range(len(self))
        )

    @property
    def provides_image(self):
        return any(v.provides_image for v in self.volumes)

    @property
    def fragments(self):
        return {
            index.fragment
            for indices in self._indices.values()
            for index in indices
            if index.fragment is not None
        }

    @property
    def provides_mesh(self):
        return any(v.provides_mesh for v in self.volumes)

    @property
    def formats(self):
        return {f for v in self.volumes for f in v.formats}

    @property
    def is_labelled(self):
        return self.maptype == MapType.LABELLED

    @property
    def affine(self):
        if self._affine_cached is None:
            # we compute the affine from a volumetric volume provider
            for fmt in _volume.Volume.SUPPORTED_FORMATS:
                if fmt not in _volume.Volume.MESH_FORMATS:
                    if fmt not in self.formats:
                        continue
                    try:
                        self._affine_cached = self.fetch(index=MapIndex(volume=0), format=fmt).affine
                        break
                    except Exception:
                        logger.debug("Caught exceptions:\n", exc_info=1)
                        continue
            else:
                raise RuntimeError(f"No volumetric provider in {self} to derive the affine matrix.")
        if not isinstance(self._affine_cached, np.ndarray):
            logger.error("invalid affine:", self._affine_cached)
        return self._affine_cached

    def __iter__(self):
        return self.fetch_iter()

    def compress(self, **kwargs):
        """
        Converts this map into a labelled 3D parcellation map, obtained by
        taking the voxelwise maximum across the mapped volumes and fragments,
        and re-labelling regions sequentially.

        Paramaters
        ----------
        **kwargs: Takes the fetch arguments of its space's template.

        Returns
        -------
        parcellationmap.Map
        """
        if len(self.volumes) == 1 and not self.fragments:
            raise RuntimeError("The map cannot be merged since there are no multiple volumes or fragments.")

        # initialize empty volume according to the template
        template = self.space.get_template().fetch(**kwargs)
        result_data = np.zeros_like(np.asanyarray(template.dataobj))
        result_affine = template.affine
        voxelwise_max = np.zeros_like(result_data)
        interpolation = 'nearest' if self.is_labelled else 'linear'
        next_labelindex = 1
        region_indices = defaultdict(list)

        for volidx in siibra_tqdm(
            range(len(self.volumes)), total=len(self.volumes), unit='maps',
            desc=f"Compressing {len(self.volumes)} {self.maptype.name.lower()} volumes into single-volume parcellation",
            disable=(len(self.volumes) == 1)
        ):
            for frag in siibra_tqdm(
                self.fragments, total=len(self.fragments), unit='maps',
                desc=f"Compressing {len(self.fragments)} {self.maptype.name.lower()} fragments into single-fragment parcellation",
                disable=(len(self.fragments) == 1 or self.fragments is None)
            ):
                mapindex = MapIndex(volume=volidx, fragment=frag)
                img = self.fetch(mapindex)
                if np.linalg.norm(result_affine - img.affine) > 1e-14:
                    logger.debug(f"Compression requires to resample volume {volidx} ({interpolation})")
                    img_data = resample_array_to_array(
                        img.get_fdata(), img.affine, result_data, result_affine
                    )
                else:
                    img_data = img.get_fdata()

                if self.is_labelled:
                    labels = set(np.unique(img_data)) - {0}
                else:
                    labels = {None}

                for label in labels:
                    with QUIET:
                        mapindex.__setattr__("label", int(label))
                        region = self.get_region(index=mapindex)
                    if region is None:
                        logger.warning(f"Label index {label} is observed in map volume {self}, but no region is defined for it.")
                        continue
                    region_indices[region.name].append({"volume": 0, "label": next_labelindex})
                    if label is None:
                        update_voxels = (img_data > voxelwise_max)
                    else:
                        update_voxels = (img_data == label)
                    result_data[update_voxels] = next_labelindex
                    voxelwise_max[update_voxels] = img_data[update_voxels]
                    next_labelindex += 1

        return Map(
            identifier=f"{create_key(self.name)}_compressed",
            name=f"{self.name} compressed",
            space_spec=self._space_spec,
            parcellation_spec=self._parcellation_spec,
            indices=region_indices,
            volumes=[_volume.from_array(
                result_data, result_affine, self._space_spec, name=self.name + " compressed"
            )]
        )

    def compute_centroids(self) -> Dict[str, point.Point]:
        """
        Compute a dictionary of the centroids of all regions in this map.

        Returns
        -------
        Dict[str, point.Point]
            Region names as keys and computed centroids as items.
        """
        centroids = {}
        maparr = None
        for regionname, indexlist in siibra_tqdm(
            self._indices.items(), unit="regions", desc="Computing centroids"
        ):
            assert len(indexlist) == 1
            index = indexlist[0]
            if index.label == 0:
                continue
            with QUIET:
                mapimg = self.fetch(index=index)  # returns a mask of the region
            maparr = np.asanyarray(mapimg.dataobj)
            centroid_vox = np.mean(np.where(maparr == 1), axis=1)
            assert regionname not in centroids
            centroids[regionname] = point.Point(
                np.dot(mapimg.affine, np.r_[centroid_vox, 1])[:3], space=self.space
            )
        return centroids

    def get_resampled_template(self, **fetch_kwargs) -> _volume.Volume:
        """
        Resample the reference space template to fetched map image. Uses
        nilearn.image.resample_to_img to resample the template.

        Parameters
        ----------
        **fetch_kwargs: takes the arguments of Map.fetch()

        Returns
        -------
        Volume
        """
        source_template = self.space.get_template().fetch()
        map_image = self.fetch(**fetch_kwargs)
        img = image.resample_to_img(source_template, map_image, interpolation='continuous')
        return _volume.from_array(
            data=img.dataobj,
            affine=img.affine,
            space=self.space,
            name=f"{source_template} resampled to coordinate system of {self}"
        )

    def colorize(self, values: dict, **kwargs) -> _volume.Volume:
        """Colorize the map with the provided regional values.

        Parameters
        ----------
        values : dict
            Dictionary mapping regions to values

        Return
        ------
        Nifti1Image
        """

        result = None
        for volidx, vol in enumerate(self.fetch_iter(**kwargs)):
            if isinstance(vol, dict):
                raise NotImplementedError("Map colorization not yet implemented for meshes.")
            img = np.asanyarray(vol.dataobj)
            maxarr = np.zeros_like(img)
            for r, value in values.items():
                index = self.get_index(r)
                if index.volume != volidx:
                    continue
                if result is None:
                    result = np.zeros_like(img)
                    affine = vol.affine
                if index.label is None:
                    updates = img > maxarr
                    result[updates] = value
                    maxarr[updates] = img[updates]
                else:
                    result[img == index.label] = value

        return _volume.from_array(
            data=result,
            affine=affine,
            space=self.space,
            name=f"Custom colorization of {self}"
        )

    def get_colormap(self, region_specs: Iterable = None):
        """
        Generate a matplotlib colormap from known rgb values of label indices.

        Parameters
        ----------
        region_specs: iterable(regions), optional
            Optional parameter to only color the desired regions.

        Returns
        -------
        ListedColormap
        """
        from matplotlib.colors import ListedColormap
        import numpy as np

        colors = {}
        if region_specs is not None:
            include_region_names = {
                self.parcellation.get_region(region_spec).name for region_spec in region_specs
            }
        else:
            include_region_names = None

        for regionname, indices in self._indices.items():
            for index in indices:
                if index.label is None:
                    continue

                if (include_region_names is not None) and (regionname not in include_region_names):
                    continue
                else:
                    region = self.get_region(index=index)
                    if region.rgb is not None:
                        colors[index.label] = region.rgb

        pallette = np.array(
            [
                list(colors[i]) + [1] if i in colors else [0, 0, 0, 0]
                for i in range(max(colors.keys()) + 1)
            ]
        ) / [255, 255, 255, 1]
        return ListedColormap(pallette)

    def sample_locations(self, regionspec, numpoints: int):
        """ Sample 3D locations inside a given region.

        The probability distribution is approximated from the region mask based
        on the squared distance transform.

        Parameters
        ----------
        regionspec: Region or str
            Region to be used
        numpoints: int
            Number of samples to draw

        Returns
        -------
        PointSet
            Sample points in physcial coordinates corresponding to this
            parcellationmap
        """
        index = self.get_index(regionspec)
        mask = self.fetch(index=index)
        arr = np.asanyarray(mask.dataobj)
        if arr.dtype.char in np.typecodes['AllInteger']:
            # a binary mask - use distance transform to get sampling weights
            W = distance_transform_edt(np.asanyarray(mask.dataobj))**2
        else:
            # a statistical map - interpret directly as weights
            W = arr
        p = (W / W.sum()).ravel()
        XYZ_ = np.array(
            np.unravel_index(np.random.choice(len(p), numpoints, p=p), W.shape)
        ).T
        XYZ = np.dot(mask.affine, np.c_[XYZ_, np.ones(numpoints)].T)[:3, :].T
        return pointset.PointSet(XYZ, space=self.space)

    def to_sparse(self):
        """
        Creates a SparseMap object from this parcellation map object.

        Returns
        -------
        SparseMap
        """
        from .sparsemap import SparseMap
        indices = {
            regionname: [
                {'volume': idx.volume, 'label': idx.label, 'fragment': idx.fragment}
                for idx in indexlist
            ]
            for regionname, indexlist in self._indices.items()
        }
        return SparseMap(
            identifier=self.id,
            name=self.name,
            space_spec={'@id': self.space.id},
            parcellation_spec={'@id': self.parcellation.id},
            indices=indices,
            volumes=self.volumes,
            shortname=self.shortname,
            description=self.description,
            modality=self.modality,
            publications=self.publications,
            datasets=self.datasets
        )

    def _read_voxel(
        self,
        x: Union[int, np.ndarray, List],
        y: Union[int, np.ndarray, List],
        z: Union[int, np.ndarray, List]
    ):
        def _read_voxels_from_volume(xyz, volimg):
            xyz = np.stack(xyz, axis=1)
            valid_points_mask = np.all([(0 <= di) & (di < vol_size) for vol_size, di in zip(volimg.shape, xyz.T)], axis=0)
            x, y, z = xyz[valid_points_mask].T
            valid_points_indices, *_ = np.where(valid_points_mask)
            valid_data_points = np.asanyarray(volimg.dataobj)[x, y, z]
            return zip(valid_points_indices, valid_data_points)

        # integers are just single-element arrays, cast to avoid an extra code branch for integers
        x, y, z = [np.array(di) for di in (x, y, z)]

        fragments = self.fragments or {None}
        return [
            (pointindex, volume, fragment, data_point)
            for fragment in fragments
            for volume, volimg in enumerate(self.fetch_iter(fragment=fragment))
            # transformations or user input might produce points outside the volume, filter these out.
            for (pointindex, data_point) in _read_voxels_from_volume((x, y, z), volimg)
        ]

    def _assign(
        self,
        item: location.Location,
        minsize_voxel=1,
        lower_threshold=0.0,
        **kwargs
    ) -> List[Union[MapAssignment, AssignImageResult]]:
        """
        For internal use only. Returns a dataclass, which provides better static type checking.
        """

        if isinstance(item, point.Point):
            return self._assign_points(
                pointset.PointSet([item], item.space, sigma_mm=item.sigma),
                lower_threshold
            )
        if isinstance(item, pointset.PointSet):
            return self._assign_points(item, lower_threshold)
        if isinstance(item, _volume.Volume):
            return self._assign_volume(
                queryvolume=item,
                lower_threshold=lower_threshold,
                minsize_voxel=minsize_voxel,
                **kwargs
            )

        raise RuntimeError(
            f"Items of type {item.__class__.__name__} cannot be used for region assignment."
        )

    def assign(
        self,
        item: location.Location,
        minsize_voxel=1,
        lower_threshold=0.0,
        **kwargs
    ) -> "pd.DataFrame":
        """Assign an input Location to brain regions.

        The input is assumed to be defined in the same coordinate space
        as this parcellation map.

        Parameters
        ----------
        item: Location
            A spatial object defined in the same physical reference space as
            this parcellation map, which could be a point, set of points, or
            image volume. If it is an image, it will be resampled to the same voxel
            space if its affine transformation differs from that of the
            parcellation map. Resampling will use linear interpolation for float
            image types, otherwise nearest neighbor.
        minsize_voxel: int, default: 1
            Minimum voxel size of image components to be taken into account.
        lower_threshold: float, default: 0
            Lower threshold on values in the statistical map. Values smaller
            than this threshold will be excluded from the assignment computation.

        Returns
        -------
        pandas.DataFrame
            A table of associated regions and their scores per component found
            in the input image, or per coordinate provided. The scores are:

                - Value: Maximum value of the voxels in the map covered by an
                input coordinate or input image signal component.
                - Pearson correlation coefficient between the brain region map
                and an input image signal component (NaN for exact coordinates)
                - Contains: Percentage of the brain region map contained in an
                input image signal component, measured from their binarized
                masks as the ratio between the volume of their intersection
                and the volume of the brain region (NaN for exact coordinates)
                - Contained: Percentage of an input image signal component
                contained in the brain region map, measured from their binary
                masks as the ratio between the volume of their intersection and
                the volume of the input image signal component (NaN for exact
                coordinates)
        """

        assignments = self._assign(item, minsize_voxel, lower_threshold, **kwargs)

        # format assignments as pandas dataframe
        columns = [
            "input structure",
            "centroid",
            "volume",
            "fragment",
            "region",
            "correlation",
            "intersection over union",
            "map value",
            "map weighted mean",
            "map containedness",
            "input weighted mean",
            "input containedness"
        ]
        if len(assignments) == 0:
            return pd.DataFrame(columns=columns)
        # determine the unique set of observed indices in order to do region lookups
        # only once for each map index occuring in the point list
        labelled = self.is_labelled  # avoid calling this in a loop
        observed_indices = {  # unique set of observed map indices. NOTE: len(observed_indices) << len(assignments)
            (
                a.volume,
                a.fragment,
                a.map_value if labelled else None
            )
            for a in assignments
        }
        region_lut = {  # lookup table of observed region objects
            (v, f, l): self.get_region(
                index=MapIndex(
                    volume=int(v),
                    label=l if l is None else int(l),
                    fragment=f
                )
            )
            for v, f, l in observed_indices
        }

        dataframe_list = []
        for a in assignments:
            item_to_append = {
                "input structure": a.input_structure,
                "centroid": a.centroid,
                "volume": a.volume,
                "fragment": a.fragment,
                "region": region_lut[
                    a.volume,
                    a.fragment,
                    a.map_value if labelled else None
                ],
            }
            # because AssignImageResult is a subclass of Assignment
            # need to check for isinstance AssignImageResult first
            if isinstance(a, AssignImageResult):
                item_to_append = {
                    **item_to_append,
                    **{
                        "correlation": a.correlation,
                        "intersection over union": a.intersection_over_union,
                        "map value": a.map_value,
                        "map weighted mean": a.weighted_mean_of_first,
                        "map containedness": a.intersection_over_first,
                        "input weighted mean": a.weighted_mean_of_second,
                        "input containedness": a.intersection_over_second,
                    }
                }
            elif isinstance(a, MapAssignment):
                item_to_append = {
                    **item_to_append,
                    **{
                        "correlation": None,
                        "intersection over union": None,
                        "map value": a.map_value,
                        "map weighted mean": None,
                        "map containedness": None,
                        "input weighted mean": None,
                        "input containedness": None,
                    }
                }
            else:
                raise RuntimeError("assignments must be of type Assignment or AssignImageResult!")

            dataframe_list.append(item_to_append)
        return (
            pd.DataFrame(dataframe_list)
            .convert_dtypes()  # convert will guess numeric column types
            .reindex(columns=columns)
            .dropna(axis='columns', how='all')
        )

    def _assign_points(self, points: pointset.PointSet, lower_threshold: float) -> List[MapAssignment]:
        """
        assign a PointSet to this parcellation map.

        Parameters:
        -----------
        lower_threshold: float, default: 0
            Lower threshold on values in the statistical map. Values smaller than
            this threshold will be excluded from the assignment computation.
        """
        assignments = []

        if points.space != self.space:
            logger.info(
                f"Coordinates will be converted from {points.space.name} "
                f"to {self.space.name} space for assignment."
            )
        # convert sigma to voxel coordinates
        scaling = np.array(
            [np.linalg.norm(self.affine[:, i]) for i in range(3)]
        ).mean()
        phys2vox = np.linalg.inv(self.affine)

        # if all points have the same sigma, and lead to a standard deviation
        # below 3 voxels, we are much faster with a multi-coordinate readout.
        if points.has_constant_sigma:
            sigma_vox = points.sigma[0] / scaling
            if sigma_vox < 3:
                pts_warped = points.warp(self.space.id)
                X, Y, Z = (np.dot(phys2vox, pts_warped.homogeneous.T) + 0.5).astype("int")[:3]
                for pointindex, vol, frag, value in self._read_voxel(X, Y, Z):
                    if value > lower_threshold:
                        position = pts_warped[pointindex].coordinate
                        assignments.append(
                            MapAssignment(
                                input_structure=pointindex,
                                centroid=tuple(np.array(position).round(2)),
                                volume=vol,
                                fragment=frag,
                                map_value=value
                            )
                        )
                return assignments

        # if we get here, we need to handle each point independently.
        # This is much slower but more precise in dealing with the uncertainties
        # of the coordinates.
        for pointindex, pt in siibra_tqdm(
            enumerate(points.warp(self.space.id)),
            total=len(points), desc="Assigning points",
        ):
            sigma_vox = pt.sigma / scaling
            if sigma_vox < 3:
                # voxel-precise - just read out the value in the maps
                N = len(self)
                logger.debug(f"Assigning coordinate {tuple(pt)} to {N} maps")
                x, y, z = (np.dot(phys2vox, pt.homogeneous) + 0.5).astype("int")[:3]
                values = self._read_voxel(x, y, z)
                for _, vol, frag, value in values:
                    if value > lower_threshold:
                        assignments.append(
                            MapAssignment(
                                input_structure=pointindex,
                                centroid=tuple(pt),
                                volume=vol,
                                fragment=frag,
                                map_value=value
                            )
                        )
            else:
                logger.debug(
                    f"Assigning uncertain coordinate {tuple(pt)} to {len(self)} maps."
                )
                kernel = create_gaussian_kernel(sigma_vox, 3)
                r = int(kernel.shape[0] / 2)  # effective radius
                assert pt.homogeneous.shape[0] == 1
                xyz_vox = (np.dot(phys2vox, pt.homogeneous.T) + 0.5).astype("int")
                shift = np.identity(4)
                shift[:3, -1] = xyz_vox[:3, 0] - r
                # build niftiimage with the Gaussian blob,
                # then recurse into this method with the image input
                gaussian_kernel = _volume.from_array(
                    data=kernel,
                    affine=np.dot(self.affine, shift),
                    space=self.space,
                    name=f"Gaussian kernel of {pt}"
                )
                for entry in self._assign(
                    item=gaussian_kernel,
                    lower_threshold=lower_threshold,
                    split_components=False
                ):
                    entry.input_structure = pointindex
                    entry.centroid = tuple(pt)
                    assignments.append(entry)
        return assignments

    def _assign_volume(
        self,
        queryvolume: "_volume.Volume",
        lower_threshold: float,
        split_components: bool = True,
        **kwargs
    ) -> List[AssignImageResult]:
        """
        Assign an image volume to this parcellation map.

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
        # TODO: split_components is not known to `assign`
        # TODO: `minsize_voxel` is not used here. Consider the implementation of `assign` again.
        if kwargs:
            logger.info(f"The keywords {[k for k in kwargs]} are not passed on during volume assignment.")

        assert queryvolume.space == self.space, ValueError("Assigned volume must be in the same space as the map.")

        if split_components:
            iter_components = lambda arr: connected_components(arr)
        else:
            iter_components = lambda arr: [(0, arr)]

        queryimg = queryvolume.fetch()
        queryimgarr = np.asanyarray(queryimg.dataobj)
        assignments = []
        all_indices = [
            index
            for regionindices in self._indices.values()
            for index in regionindices
        ]
        with QUIET and provider.SubvolumeProvider.UseCaching():
            for index in siibra_tqdm(
                all_indices,
                desc=f"Assigning {queryvolume} to {self}",
                disable=len(all_indices) < 5,
                unit="map",
                leave=False
            ):
                region_map = self.fetch(index=index)
                region_map_arr = np.asanyarray(region_map.dataobj)
                # the shape and affine are checked by `nilearn.image.resample_to_img()`
                # and returns the original data if resampling is not necessary.
                queryimgarr_res = resample_array_to_array(
                    queryimgarr,
                    queryimg.affine,
                    region_map_arr,
                    region_map.affine
                )
                for compmode, voxelmask in iter_components(queryimgarr_res):
                    scores = compare_arrays(
                        voxelmask,
                        region_map.affine,  # after resampling, both should have the same affine
                        region_map_arr,
                        region_map.affine
                    )
                    component_position = np.array(np.where(voxelmask)).T.mean(0)
                    if scores.intersection_over_union > lower_threshold:
                        assignments.append(
                            AssignImageResult(
                                input_structure=compmode,
                                centroid=tuple(component_position.round(2)),
                                volume=index.volume,
                                fragment=index.fragment,
                                map_value=index.label,
                                **asdict(scores)
                            )
                        )

        return assignments


def from_volume(
    name: str,
    volume: Union[_volume.Volume, List[_volume.Volume]],
    regionnames: List[str],
    regionlabels: List[int],
    parcellation_spec: Union[str, "parcellation.Parcellation"] = None
) -> 'Map':
    """
    Add a custom labelled parcellation map to siibra from a labelled NIfTI file.

    Parameters:
    ------------
    name: str
        Human-readable name of the parcellation.
    volume: Volume, or a list of Volumes.
    space_spec: str, Space
        Specification of the reference space (space object, name, keyword, or id - e.g. 'mni152').
    regionnames: list[str]
        List of human-readable names of the mapped regions.
    regionlabels: list[int]
        List of integer labels in the nifti file corresponding to the list of regions.
    parcellation: str or Parcellation. Optional.
        If the related parcellation already defined or preconfigured in siibra.
    """
    # providers and map indices
    providers = []
    volumes = volume if isinstance(volume, list) else [volume]
    map_space = volumes[0].space
    assert all(v.space == map_space for v in volumes), "Volumes have to be in the same space"
    for vol_idx, vol in enumerate(volumes):
        image = vol.fetch()
        arr = np.asanyarray(image.dataobj)
        labels_in_volume = np.unique(arr)[1:].astype('int')

        # populate region indices from given name/label lists
        indices = dict()
        for label, regionname in zip(regionlabels, regionnames):
            if label not in labels_in_volume:
                logger.warning(
                    f"Label {label} not mapped in the provided NIfTI volume -> "
                    f"region '{regionname} will not be in the map."
                )
            elif label in [v[0]['label'] for v in indices.values() if v[0]['volume'] == vol_idx]:
                logger.warning(f"Label {label} already defined in the same volume; will not map it to '{regionname}'.")
            else:
                assert regionname not in indices, f"'{regionname}' must be unique in `regionnames`."
                indices[regionname] = [{'volume': vol_idx, 'label': label}]

        # check for any remaining labels in the NIfTI volume
        unnamed_labels = list(set(labels_in_volume) - set(regionlabels))
        if unnamed_labels:
            logger.warning(
                f"The following labels appear in the NIfTI volume {vol_idx}, but not in "
                f"the specified regions: {', '.join(str(l) for l in unnamed_labels)}. "
                "They will be removed from the nifti volume."
            )
            for label in unnamed_labels:
                arr[arr == label] = 0
        providers.extend(vol._providers.values())

    # parcellation
    if parcellation_spec is None:
        parcellation_spec = name
    try:
        parcobj = parcellation.Parcellation.registry().get(parcellation_spec)
        logger.info(f"Using '{parcellation_spec}', siibra decoded the parcellation as '{parcobj}'")
    except Exception:
        logger.info(
            f"Using '{parcellation_spec}', siibra could not decode the "
            " parcellation. Building a new parcellation."
        )
        # build a new parcellation
        parcobj = parcellation.Parcellation(
            identifier=generate_uuid(','.join(regionnames)),
            name=name,
            species=vol.space.species,
            regions=list(map(_region.Region, regionnames)),
        )
        if parcobj.key not in list(parcellation.Parcellation.registry()):
            parcellation.Parcellation.registry().add(parcobj.key, parcobj)

    for region in siibra_tqdm(
        indices.keys(),
        desc="Checking if provided regions are defined in the parcellation."
    ):
        try:
            _ = parcobj.get_region(region)
        except Exception:
            logger.warning(f"'{region}' is missing in the parcellation.")

    # build the parcellation map object
    parcmap = Map(
        identifier=generate_uuid(name),
        name=f"{name} map in {map_space.name}",
        space_spec={"@id": map_space.id},
        parcellation_spec={'name': parcobj.name},
        indices=indices,
        volumes=volumes
    )

    # add it to siibra's registry
    Map.registry().add(parcmap.key, parcmap)

    # return the map - note that it has a pointer to the parcellation
    return parcmap
