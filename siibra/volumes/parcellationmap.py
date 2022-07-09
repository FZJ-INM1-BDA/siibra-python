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

from .volume import VolumeSrc, ImageProvider
from .util import create_gaussian_kernel, argmax_dim4

from .. import logger, QUIET
from ..commons import ParcellationIndex, MapType, compare_maps
from ..core.space import Point, PointSet, Space, BoundingBox
from ..core.region import Region
from ..retrieval import CACHE, GitlabConnector

import numpy as np
from nibabel import Nifti1Image, funcs, load
from nilearn import image
from memoization import cached
from tqdm import tqdm
from abc import abstractmethod, ABC
from typing import Union
from os import path
from numbers import Number
import pandas as pd
from math import ceil, log10
import gzip
from scipy.ndimage.morphology import distance_transform_edt


class ParcellationMap(ABC):
    """
    Represents a brain map in a particular reference space, with
    explicit knowledge about the region information per labelindex or channel.
    """

    _instances = {}
    _regions_cached = None
    _maploaders_cached = None

    def __init__(self, parcellation, space: Space, maptype=MapType):
        """
        Construct a ParcellationMap for the given parcellation and space.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        maptype : MapType
            The desired type of the map
        """
        if not parcellation.supports_space(space):
            raise ValueError(
                'Parcellation "{}" does not provide a map for space "{}"'.format(
                    parcellation.name, space.name
                )
            )

        if isinstance(maptype, str):
            self.maptype = MapType[maptype.upper()]
        else:
            self.maptype = maptype
        self.parcellation = parcellation
        self.space = space

    @classmethod
    def get_instance(cls, parcellation, space: Space, maptype: MapType):
        """
        Returns the ParcellationMap object of the requested type.
        """
        key = (parcellation.key, space.key, maptype)

        # If an instance is already available, return it
        if key in cls._instances:
            return cls._instances[key]

        # Otherwise, create a new object
        if space.type == "gii":
            classes = {
                MapType.LABELLED: LabelledSurface,
                MapType.CONTINUOUS: None,
            }
        else:
            classes = {
                MapType.LABELLED: LabelledParcellationVolume,
                MapType.CONTINUOUS: ContinuousParcellationVolume,
            }
        if maptype in classes:
            instance = classes[maptype](parcellation, space)
        elif maptype is None:
            logger.warning(
                "No maptype provided when requesting the parcellation map. Falling back to MapType.LABELLED"
            )
            instance = classes[MapType.LABELLED](parcellation, space)
        else:
            raise ValueError(
                f"Cannote create a map of type '{maptype}' - this is an unkown type."
            )

        if (instance is None) or (len(instance) == 0):
            raise ValueError(
                f"No data found to construct a {maptype} map for {parcellation.name} in {space.name}."
            )

        cls._instances[key] = instance
        return instance

    @property
    def maploaders(self):
        if self._maploaders_cached is None:
            self._define_maps_and_regions()
        return self._maploaders_cached

    @property
    def regions(self):
        """
        Dictionary of regions associated to the parcellion map, indexed by ParcellationIndex.
        Lazy implementation - self._link_regions() will be called when the regions are accessed for the first time.
        """
        if self._regions_cached is None:
            self._define_maps_and_regions()
        return self._regions_cached

    @property
    def names(self):
        return self.parcellation.names

    @abstractmethod
    def _define_maps_and_regions(self):
        """
        implemented by derived classes, to produce the lists _regions_cached and _maploaders_cached.
        The first is a dictionary indexed by ParcellationIndex,
        the latter a list of functions for loading the different maps.
        """
        pass

    def fetch_iter(self, resolution_mm=None, voi: BoundingBox = None, variant=None):
        """
        Returns an iterator to fetch all available maps sequentially.

        Parameters
        ----------
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes.
            If None, the smallest possible resolution will be chosen.
            If -1, the largest feasible resolution will be chosen.
        variant : str
            Optional specification of variant of the maps. For example,
            fsaverage provides the 'pial', 'white matter' and 'inflated' surface variants.
        """
        logger.debug(f"Iterator for fetching {len(self)} parcellation maps")
        return (
            fnc(res=resolution_mm, voi=voi, variant=variant) for fnc in self.maploaders
        )

    def fetch(
        self,
        mapindex: int = 0,
        resolution_mm: float = None,
        voi: BoundingBox = None,
        variant=None,
    ):
        """
        Fetches one particular map.

        Parameters
        ----------
        mapindex : int
            The index of the available maps to be fetched.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes.
            If None, the smallest possible resolution will be chosen.
            If -1, the largest feasible resolution will be chosen.
        variant : str
            Optional specification of a specific variant to use for the maps. For example,
            fsaverage provides the 'pial', 'white matter' and 'inflated' surface variants.
        """
        if mapindex < len(self):
            if len(self) > 1:
                logger.info(
                    f"Returning map {mapindex+1} of in total {len(self)} available maps."
                )
            return self.maploaders[mapindex](
                res=resolution_mm, voi=voi, variant=variant
            )
        else:
            raise ValueError(
                f"'{len(self)}' maps available, but a mapindex of {mapindex} was requested."
            )

    def __len__(self):
        """
        Returns the number of maps available in this parcellation.
        """
        return len(self.maploaders)

    def __contains__(self, spec):
        """
        Test if a map identified by the given specification is included in this parcellation map.
        For integer values, it is checked wether a corresponding slice along the fourth dimension could be extracted.
        Alternatively, a region object can be provided, and it will be checked wether the region is mapped.
        You might find the decode_region() function of Parcellation and Region objects useful for the latter.
        """
        if isinstance(spec, int):
            return spec in range(len(self.maploaders))
        elif isinstance(spec, Region):
            for _, region in self.regions.items():
                if region == spec:
                    return True
        return False

    def decode_region(self, regionspec: Union[str, Region]):
        """
        Given a unique specification, return the corresponding region
        that is mapped in this ParcellationMap.
        The spec could be a label index, a (possibly incomplete) name, or a
        region object.
        This method is meant to definitely determine a valid region. Therefore,
        if no match is found, it raises a ValueError.

        Parameters
        ----------
        regionspec : any of
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key,
            - an integer, which is interpreted as a labelindex,
            - a region object
            - a full ParcellationIndex

        Return
        ------
        Region object
        """
        # make sure we have a region object that matches the parcellation
        if isinstance(regionspec, Region):
            region = regionspec
        else:
            region = self.parcellation.decode_region(regionspec)

        if region in self.regions.values():
            # a perfect match
            return region

        # If the given region is not directly found in the map,
        # see if there is a unique match among its children.
        matches = [
            (r, len(r.path)) for r in region
            if (r in self.parcellation) and (r in self.regions.values())
        ]
        if len(matches) == 0:
            raise IndexError(f"Region '{region.name}' is not mapped in {str(self)}.")
        mindepth = min(m[1] for m in matches)
        candidates = list(filter(lambda v: v[1] == mindepth, matches))

        if len(candidates) == 1:
            return candidates[0][0]
        elif len(candidates) == 0:
            raise IndexError(f"Region '{region.name}' is not mapped in {str(self)}.")
        else:
            raise IndexError(
                f"Ambiguous assignment of '{region.name}' to parcellation map, "
                f"it resolves to {', '.join(c.name for c,d in candidates)} "
                f"in {str(self)}."
            )

    def decode_index(self, mapindex=None, labelindex=None):
        """
        Returns the region associated with a particular parcellation index.

        Parameters
        ----------
        mapindex : Sequential index of the 3D map used, if more than one are included
        labelindex : Label index of the region, if the map is a labelled volume
        """
        pindex = ParcellationIndex(map=mapindex, label=labelindex)
        region = self.regions.get(pindex)
        if region is None:
            raise ValueError(f"Could not decode parcellation index {pindex}")
        else:
            return region

    def get_index(self, regionspec: Union[str, Region]):
        """
        Find the ParcellationIndex for a given region.

        Parameters
        ----------
        regionspec : str or Region
            Partial name of region, or Region object

        Return
        ------
        list of MapIndex objects
        """
        region = (
            self.parcellation.decode_region(regionspec)
            if isinstance(regionspec, str)
            else regionspec
        )
        subregions = []
        for idx, r in self.regions.items():
            if r == region:
                return [idx]
            elif r.has_parent(region):
                subregions.append((idx, r))
        if len(subregions) == 0:
            raise IndexError(
                f"Could not decode region specified by {regionspec} in {self.parcellation.name}"
            )

        # if we found maps of child regions, we want the mapped leaves to be identical to the leaves of the requested region.
        children_found = {c for _, r in subregions for c in r.leaves}
        children_requested = {c for c in region.leaves}
        if children_found != children_requested:
            raise IndexError(
                f"Cannot decode {regionspec} for the map in {self.space.name}, as it seems only partially mapped there."
            )
        return [idx for idx, _ in subregions]


class ParcellationVolume(ParcellationMap, ImageProvider):
    """
    Represents a brain map in a particular volumetric reference space, with
    explicit knowledge about the region information per labelindex or channel.

    There are two types:
        1) Parcellation maps / labelled volumes (MapType.LABELLED)
            A 3D or 4D volume with integer labels separating different,
            non-overlapping regions. The number of regions corresponds to the
            number of nonzero image labels in the volume.
        2) 4D overlapping regional maps (often probability maps) (MapType.CONTINUOUS)
            a 4D volume where each "time"-slice is a 3D volume representing
            a map of a particular brain region. This format is used for
            probability maps and similar continuous forms. The number of
            regions correspond to the z dimension of the 4 object.

    ParcellationMaps can be also constructred from neuroglancer (BigBrain) volumes if
    a feasible downsampled resolution is provided.
    """

    # Which types of available volumes should be preferred if multiple choices are available?
    PREFERRED_VOLUMETYPES = ["nii", "neuroglancer/precomputed"]

    _regions_cached = None
    _maploaders_cached = None

    def __init__(self, parcellation, space: Space, maptype=MapType):
        """
        Construct a ParcellationMap for the given parcellation and space.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        maptype : MapType
            The desired type of the map
        """
        ParcellationMap.__init__(self, parcellation, space, maptype)

    def fetch_all(self):
        """Returns a 4D array containing all 3D maps.

        All available maps are stacked along the 4th dimension.
        Note that this can be quite memory-intensive for continuous maps.
        If you just want to iterate over maps, prefer using
            'for img in ParcellationMaps.fetch_iter():'
        """
        N = len(self)
        with QUIET:
            im0 = self.fetch(mapindex=0)
        out_shape = (N,) + im0.shape
        logger.info(f"Create 4D array from {N} maps with size {im0.shape + (N,)}")
        out_data = np.empty(out_shape, dtype=im0.dataobj.dtype)

        for mapindex, img in tqdm(
            enumerate(self.fetch_iter()), total=N, disable=logger.level > 20
        ):
            out_data[mapindex] = np.asanyarray(img.dataobj)

        return funcs.squeeze_image(
            Nifti1Image(np.rollaxis(out_data, 0, out_data.ndim), im0.affine)
        )

    def fetch_relabelled(self):
        """
        Returns a relabelled 3D parcellation volume, obtained by taking the
        maximum across maps at each voxel and labelling regions sequentially.
        """
        result = None
        maxarr = None
        regions = {}
        new_labelindex = 1

        for mapindex in tqdm(
            range(len(self)), total=len(self), unit='maps'
        ):

            with QUIET:
                mapimg = self.fetch(mapindex=mapindex)
            maparr = np.asanyarray(mapimg.dataobj)

            if result is None:
                lblarr = np.zeros_like(maparr)
                maxarr = np.zeros_like(maparr)
                result = Nifti1Image(lblarr, mapimg.affine)

            if self.maptype == MapType.LABELLED:
                labels = set(np.unique(maparr)) - {0}
            else:
                labels = {None}

            for labelindex in labels:
                region = self.parcellation.decode_region(ParcellationIndex(mapindex, labelindex))
                if labelindex is None:
                    updates = (maparr > maxarr)
                else:
                    updates = (maparr == labelindex)

                lblarr[updates] = new_labelindex
                maxarr[updates] = maparr[updates]
                regions[new_labelindex] = region.name
                new_labelindex += 1

        return result, regions

    def compute_centroids(self):
        """Compute a dictionary of the centroids of all regions in this map.
        """
        centroids = {}
        # list of regions sorted by mapindex
        regions = sorted(self.regions.items(), key=lambda v: (v[0].map, v[0].label))
        current_mapindex = -1
        maparr = None
        for pind, region in tqdm(regions, unit="regions", desc="Computing centroids"):
            if pind.label == 0:
                continue
            if pind.map != current_mapindex:
                current_mapindex = pind.map
                with QUIET:
                    mapimg = self.fetch(pind.map)
                maparr = np.asanyarray(mapimg.dataobj)
            if pind.label is None:
                # should be a continous map then
                assert self.maptype == MapType.CONTINUOUS
                centroid_vox = np.array(np.where(maparr > 0)).mean(1)
            else:
                centroid_vox = np.array(np.where(maparr == pind.label)).mean(1)
            assert region not in centroids
            centroids[region] = Point(
                np.dot(mapimg.affine, np.r_[centroid_vox, 1])[:3], space=self.space
            )
        return centroids

    def fetch_regionmap(
        self,
        regionspec: Union[str, int, Region],
        resolution_mm=None,
        voi: BoundingBox = None,
    ):
        """
        Extract the mask for one particular region.
        For multi-regions, returns the voxelwise maximum of their children's masks.

        Parameters
        ----------
        regionspec : labelindex, partial region name, or Region
            The desired region.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes.
            If None, the smallest possible resolution will be chosen.
            If -1, the largest feasible resolution will be chosen.

        Return
        ------
        Nifti1Image, if found, otherwise None
        """
        try:
            indices = self.get_index(regionspec)
        except IndexError:
            return None
        data = None
        affine = None
        for index in indices:
            with QUIET:
                mapimg = self.fetch(
                    resolution_mm=resolution_mm, mapindex=index.map, voi=voi
                )
            if index.label is None:  # region is defined by the whole map
                newdata = mapimg.get_fdata()
            else:  # region is defined by a particular label
                newdata = (mapimg.get_fdata() == index.label).astype(np.uint8)
            if data is None:
                data = newdata
                affine = mapimg.affine
            else:
                data = np.maximum(data, newdata)

        return Nifti1Image(data, affine)

    def get_shape(self, resolution_mm=None):
        return list(self.space.get_template().get_shape()) + [len(self)]

    def is_float(self):
        return self.maptype == MapType.CONTINUOUS

    def _load_regional_map(
        self, region: Region, resolution_mm, voi: BoundingBox = None
    ):
        logger.debug(f"Loading regional map for {region.name} in {self.space.name}")
        with QUIET:
            rmap = region.get_regional_map(self.space, self.maptype).fetch(
                resolution_mm=resolution_mm, voi=voi
            )
        return rmap


class LabelledParcellationVolume(ParcellationVolume):
    """
    Represents a brain map in a reference space, with
    explicit knowledge about the region information per labelindex or channel.
    Contains a Nifti1Image object as the "image" member.

    This form defines parcellation maps / labelled volumes (MapType.LABELLED),
    A 3D or 4D volume with integer labels separating different,
    non-overlapping regions. The number of regions corresponds to the
    number of nonzero image labels in the volume.
    """

    def __init__(self, parcellation, space: Space):
        """
        Construct a ParcellationMap for the given parcellation and space.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        """
        super().__init__(parcellation, space, MapType.LABELLED)

    def _define_maps_and_regions(self):

        self._maploaders_cached = []
        self._regions_cached = {}

        # check if the parcellation has any volumes in the requested space
        for volumetype in self.PREFERRED_VOLUMETYPES:
            sources = []
            for vsrc in self.parcellation.get_volumes(self.space.id):
                if vsrc.__class__.volume_type == volumetype:
                    sources.append(vsrc)
            if len(sources) > 0:
                break

        # Try to generate maps from suitable volume sources
        for source in sources:

            if source.volume_type != self.space.type:
                continue

            self._maploaders_cached.append(
                lambda res=None, voi=None, variant=None, s=source: self._load_map(
                    s, resolution_mm=res, voi=voi
                )
            )
            # load map at lowest resolution to map label indices to regions
            mapindex = len(self._maploaders_cached) - 1
            with QUIET:
                m = self._maploaders_cached[mapindex](res=None, voi=None, variant=None)
            unmatched = []
            for labelindex in np.unique(m.get_fdata()).astype('int'):
                if labelindex != 0:
                    pindex = ParcellationIndex(map=mapindex, label=labelindex)
                    try:
                        region = self.parcellation.decode_region(pindex)
                        if labelindex > 0:
                            self._regions_cached[pindex] = region
                        else:
                            unmatched.append(pindex)
                    except ValueError:
                        unmatched.append(pindex)
            if len(unmatched) > 0:
                logger.warning(
                    f"{len(unmatched)} parcellation indices in labelled volume couldn't be matched to region definitions in {self.parcellation.name}"
                )

        # If no maps can be generated from volume sources, try to build a collection of regional maps
        if len(self) == 0:
            self._maploaders_cached.append(
                lambda res=None, voi=None, variant=None: self._collect_maps(
                    resolution_mm=res, voi=voi
                )
            )
            # load map at lowest resolution to map label indices to regions
            m = self._maploaders_cached[0](res=None, voi=None, variant=None)

        # By now, we should have been able to generate some maps
        if len(self) == 0:
            raise RuntimeError(
                f"No maps found for {self.parcellation.name} in {self.space.name}"
            )

    @cached
    def _load_map(self, volume: VolumeSrc, resolution_mm: float, voi: BoundingBox):
        m = volume.fetch(resolution_mm=resolution_mm, voi=voi)
        if len(m.dataobj.shape) == 4:
            if m.dataobj.shape[3] == 1:
                m = Nifti1Image(dataobj=np.asarray(m.dataobj).astype(int).squeeze(), affine=m.affine)
            else:
                logger.info(
                    f"{m.dataobj.shape[3]} continuous maps given - using argmax to generate a labelled volume. "
                )
                m = argmax_dim4(m)
        if m.dataobj.dtype.kind == "f":
            logger.warning(
                f"Floating point image type encountered when building a labelled volume for {self.parcellation.name}, converting to integer."
            )
            m = Nifti1Image(dataobj=np.asarray(m.dataobj, dtype=int), affine=m.affine)
        return m

    @cached
    def _collect_maps(self, resolution_mm, voi):
        """
        Build a 3D volume from the list of available regional maps.
        Label indices will just be sequentially assigned.

        Return
        ------
        Nifti1Image, or None if no maps are found.

        """
        m = None

        # generate empty mask covering the template space
        tpl = self.space.get_template().fetch(resolution_mm, voi=voi)
        m = None

        # collect all available region maps
        regions = []
        for r in self.parcellation.regiontree:
            with QUIET:
                regionmap = r.get_regional_map(self.space, MapType.LABELLED)
            if regionmap is not None:
                regions.append(r)

        if len(regions) == 0:
            raise RuntimeError(
                f"No regional maps could be collected for {self.parcellation.name} in space {self.space.name}"
            )

        logger.info(
            f"Building labelled parcellation volume for {self.parcellation.name} "
            f"in '{self.space.name}' from {len(regions)} regional maps."
        )
        largest_label = max(self.parcellation.labels)
        next_label = ceil(log10(largest_label))
        for region in tqdm(
            regions,
            total=len(regions),
            desc=f"Collecting {len(regions)} maps",
            unit="maps",
            disable=logger.level > 20,
        ):

            # load region mask
            mask_ = self._load_regional_map(
                region, resolution_mm=resolution_mm, voi=voi
            )
            if not mask_:
                continue
            if np.prod(mask_.shape) == 0:
                continue
            # build up the aggregated mask with labelled indices
            if mask_.shape != tpl.shape:
                mask = image.resample_to_img(mask_, tpl, interpolation="nearest")
            else:
                mask = mask_

            if m is None:
                m = Nifti1Image(
                    np.zeros_like(tpl.dataobj, dtype=mask.dataobj.dtype), tpl.affine
                )
            if region.index.label is None:
                label = next_label
                next_label += 1
            else:
                label = region.index.label
            m.dataobj[mask.dataobj > 0] = label
            self._regions_cached[ParcellationIndex(map=0, label=label)] = region

        return m

    def colorize(self, values: dict):
        """Colorize the map with the provided regional values.

        Parameters
        ----------
        values : dict
            Dictionary mapping regions to values

        Return
        ------
        Nifti1Image
        """

        # generate empty image
        maps = {}
        result = None

        for region, value in values.items():
            try:
                indices = self.get_index(region)
            except IndexError:
                continue
            for index in indices:
                if index.map not in maps:
                    # load the map
                    maps[index.map] = self.fetch(index.map)
                thismap = maps[index.map]
                if result is None:
                    # create the empty output
                    result = np.zeros_like(thismap.get_fdata())
                    affine = thismap.affine
                result[thismap.get_fdata() == index.label] = value

        return Nifti1Image(result, affine)

    @cached
    def assign_coordinates(
        self, point: Union[Point, PointSet], sigma_mm=None, sigma_truncation=None
    ):
        """
        Assign regions to a physical coordinates with optional standard deviation.

        Parameters
        ----------
        point : Point or PointSet
        sigma_mm : Not needed for labelled parcellation maps
        sigma_truncation : Not needed for labelled parcellation maps
        """

        if point.space != self.space:
            logger.info(
                f"Coordinates will be converted from {point.space.name} "
                f"to {self.space.name} space for assignment."
            )

        # Convert input to Nx4 list of homogenous coordinates
        if isinstance(point, Point):
            coords = [point.warp(self.space).homogeneous]
        elif isinstance(point, PointSet):
            pointset = point
            coords = [p.homogeneous for p in pointset.warp(self.space)]
        else:
            raise ValueError("assign_coordinates expects a Point or PointSet object.")

        assignments = []
        N = len(self)
        msg = f"Assigning {len(coords)} points to {N} maps"
        assignments = [[] for _ in coords]
        for mapindex, loadfnc in tqdm(
            enumerate(self.maploaders),
            total=len(self),
            desc=msg,
            unit=" maps",
            disable=logger.level > 20,
        ):
            lmap = loadfnc()
            p2v = np.linalg.inv(lmap.affine)
            A = lmap.get_fdata()
            for i, coord in enumerate(coords):
                x, y, z = (np.dot(p2v, coord) + 0.5).astype("int")[:3]
                label = A[x, y, z]
                if label > 0:
                    region = self.decode_index(mapindex=mapindex, labelindex=label)
                    assignments[i].append((region, lmap, None))

        return assignments

    def sample_locations(self, regionspec, numpoints: int):
        """ Sample 3D locations inside a given region.

        The probability distribution is approximated from the region mask
        based on the squared distance transform.

        regionspec: valid region specification
            Region to be used
        numpoints: int
            Number of samples to draw

        Return
        ------
        samples : PointSet in physcial coordinates corresponding to this parcellationmap.

        """
        indices = self.get_index(regionspec)
        assert len(indices) > 0

        # build region mask
        B = None
        lmap = None
        for index in indices:
            lmap = self.fetch(index.map)
            M = np.asanyarray(lmap.dataobj)
            if B is None:
                B = np.zeros_like(M)
            B[M == index.label] = 1

        D = distance_transform_edt(B)**2
        p = (D / D.sum()).ravel()
        XYZ_ = np.array(
            np.unravel_index(np.random.choice(len(p), numpoints, p=p), D.shape)
        ).T
        XYZ = np.dot(lmap.affine, np.c_[XYZ_, np.ones(numpoints)].T)[:3, :].T
        return PointSet(XYZ, space=self.space)

    def assign(self, img: Nifti1Image, msg=None, quiet=False):
        """
        Assign the region of interest represented by a given volumetric image to brain regions in this map.

        TODO unify this with the corresponding methond in ContinuousParcellationMap

        Parameters:
        -----------
        img : Nifti1Image
            The input region of interest, typically a binary mask or statistical map.
        msg : str, default:None
            Message to display with the progress bar
        quiet: Boolen, default:False
            If true, no progess indicator will be displayed
        """

        if msg is None and not quiet:
            msg = f"Assigning structure to {len(self.regions)} regions"

        # How to visualize progress from the iterator?
        def plain_progress(f):
            return f

        def visual_progress(f):
            return tqdm(
                f,
                total=len(self.regions),
                desc=msg,
                unit="regions",
                disable=logger.level > 20,
            )

        progress = plain_progress if quiet else visual_progress

        # setup assignment loop
        values = {}
        pmaps = {}

        for index, region in progress(self.regions.items()):

            this = self.maploaders[index.map]()
            if not this:
                logger.warning(f"Could not load regional map for {region.name}")
                continue
            if (index.label is not None) and (index.label > 0):
                with QUIET:
                    this = region.build_mask(self.space, maptype=self.maptype)
            scores = compare_maps(img, this)
            if scores["overlap"] > 0:
                assert region not in pmaps
                pmaps[region] = this
                values[region] = scores

        assignments = [
            (region, region.index.map, scores)
            for region, scores in sorted(
                values.items(),
                key=lambda item: abs(item[1]["correlation"]),
                reverse=True,
            )
        ]
        return assignments


class ContinuousParcellationVolume(ParcellationVolume):
    """A sparse representation of list of continuous (e.g. probabilistic) brain region maps.

    It represents the 3D continuous maps of N brain regions by two data structures:
        1) 'spatial_index', a 3D volume where non-negative values represent unique
            indices into a list of region assignments
        2) 'probs', a list of region assignments where each entry is a dict

    More precisely, given
        i = spatial_index[x, y, z]
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

    def __init__(self, parcellation, space):

        ParcellationMap.__init__(self, parcellation, space, maptype="continuous")

    def _define_maps_and_regions(self):

        # Check for available maps and brain regions.
        # First look for a 4D array where the last dimension are the different maps
        self._maploaders_cached = []
        self._regions_cached = {}
        self._map4d = None
        for v in self.parcellation.volumes:
            if (
                isinstance(v, ImageProvider)
                and v.is_float()
                and v.is_4D()
                and v.get_shape()[3] > 1
            ):
                self._map4d = v.fetch()
                print(self._map4d.shape)
                for mapindex in range(self._map4d.shape[3]):
                    self._maploaders_cached.append(
                        lambda m=mapindex: self._map4d.slicer[:, :, :, m]
                    )
                    # TODO this might not be correct for parcellations other than DifumoXX
                    r = self.parcellation.decode_region(mapindex + 1)
                    self._regions_cached[
                        ParcellationIndex(map=mapindex, label=None)
                    ] = r

        if self._map4d is None:
            # No 4D array, look for regional continuous maps stored in the region tree.
            mapindex = 0
            for r in self.parcellation.regiontree.leaves:
                if r in self.regions.values():
                    continue
                if r.has_regional_map(self.space, self.maptype):
                    regionmap = r.get_regional_map(self.space, self.maptype)
                    self._maploaders_cached.append(lambda r=regionmap: r.fetch())
                    self._regions_cached[
                        ParcellationIndex(map=mapindex, label=None)
                    ] = r
                    mapindex += 1

        # either load or build the sparse index
        if not self._load_index():
            self._build_index()
            self._store_index()
        assert self.spatial_index.max() == len(self.probs) - 1

    def _load_index(self):

        self.spatial_index = None
        self.probs = []
        self.bboxes = []
        self.affine = None

        prefix = f"{self.parcellation.id}_{self.space.id}_{self.maptype}_index"
        probsfile = CACHE.build_filename(f"{prefix}", suffix="probs.txt.gz")
        bboxfile = CACHE.build_filename(f"{prefix}", suffix="bboxes.txt.gz")
        indexfile = CACHE.build_filename(f"{prefix}", suffix="index.nii.gz")

        # check if precomputed index files are available in the local cache, or on gitlab
        conn = None
        for fname in [probsfile, bboxfile, indexfile]:
            if path.isfile(fname):
                continue  # already in local cache
            if conn is None:
                conn = GitlabConnector(self._GITLAB_SERVER, self._GITLAB_PROJECT, 'main')
                files = conn.search_files()
            bname = path.basename(fname)
            if bname in files:
                logger.debug(f"Retrieving precomputed index for {self.parcellation.name}")
                raw = conn.get(bname, decode_func=lambda b: b)
                with open(fname, 'wb') as f:
                    f.write(raw)
                continue
            # if we get here, a precomputed file is not available. We have to rebuild the index.
            logger.info(f"{bname} not precomputed, need to build index.")
            return False

        indeximg = load(indexfile)
        self.spatial_index = np.asanyarray(indeximg.dataobj)
        self.affine = indeximg.affine

        with gzip.open(probsfile, "rt") as f:
            lines = f.readlines()
            msg = f"Loading spatial index for {len(self)} continuous maps"
            for line in tqdm(
                lines,
                total=len(lines),
                desc=msg,
                unit="voxels",
                disable=logger.level > 20,
            ):
                fields = line.strip().split(" ")
                mapindices = list(map(int, fields[0::2]))
                values = list(map(float, fields[1::2]))
                D = dict(zip(mapindices, values))
                self.probs.append(D)

        with gzip.open(bboxfile, "rt") as f:
            for line in f:
                fields = line.strip().split(" ")
                self.bboxes.append(
                    {
                        "minpoint": tuple(map(int, fields[:3])),
                        "maxpoint": tuple(map(int, fields[3:])),
                    }
                )

        return True

    def _store_index(self):
        # store spatial index and probability list to file
        prefix = f"{self.parcellation.id}_{self.space.id}_{self.maptype}_index"
        probsfile = CACHE.build_filename(f"{prefix}", suffix="probs.txt.gz")
        bboxfile = CACHE.build_filename(f"{prefix}", suffix="bboxes.txt.gz")
        indexfile = CACHE.build_filename(f"{prefix}", suffix="index.nii.gz")

        Nifti1Image(self.spatial_index, self.affine).to_filename(indexfile)

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

    def _build_index(self):
        """Load map image with the given index."""

        logger.info(
            f"Creating the spatial index for {self.parcellation.name} continuous maps "
            f"in {self.space.name}. This will take a minute, but is only performed once."
        )

        self.probs = []
        self.bboxes = []
        self.spatial_index = None
        self.affine = None
        for mapindex in tqdm(
            range(len(self)),
            total=len(self),
            unit="maps",
            desc=f"Fetching {len(self)} volumetric maps",
            disable=logger.level > 20,
        ):
            with QUIET:
                # retrieve the probability map
                img = self._maploaders_cached[mapindex]()

            if self.spatial_index is None:
                self.spatial_index = np.zeros(img.shape, dtype=np.int32) - 1
                self.affine = img.affine
            else:
                assert img.shape == self.shape
                assert (img.affine - self.affine).sum() == 0

            imgdata = np.asanyarray(img.dataobj)
            X, Y, Z = [v.astype("int32") for v in np.where(imgdata > 0)]
            for x, y, z, prob in zip(X, Y, Z, imgdata[X, Y, Z]):
                coord_id = self.spatial_index[x, y, z]
                if coord_id >= 0:
                    # Coordinate already seen. Add observed value.
                    assert mapindex not in self.probs[coord_id]
                    assert len(self.probs) > coord_id
                    self.probs[coord_id][mapindex] = prob
                else:
                    # New coordinate. Append entry with observed value.
                    coord_id = len(self.probs)
                    self.spatial_index[x, y, z] = coord_id
                    self.probs.append({mapindex: prob})

            self.bboxes.append(
                {
                    "minpoint": (X.min(), Y.min(), Z.min()),
                    "maxpoint": (X.max(), Y.max(), Z.max()),
                }
            )

    @property
    def shape(self):
        return self.spatial_index.shape

    def _coords(self, mapindex):
        # Nx3 array with x/y/z coordinates of the N nonzero values of the given mapindex
        coord_ids = [i for i, l in enumerate(self.probs) if mapindex in l]
        x0, y0, z0 = self.bboxes[mapindex]["minpoint"]
        x1, y1, z1 = self.bboxes[mapindex]["maxpoint"]
        return (
            np.array(
                np.where(
                    np.isin(
                        self.spatial_index[x0: x1 + 1, y0: y1 + 1, z0: z1 + 1],
                        coord_ids,
                    )
                )
            ).T
            + (x0, y0, z0)
        ).T

    def _mapped_voxels(self, mapindex):
        # returns the x, y, and z coordinates of nonzero voxels for the map
        # with the given index, together with their corresponding values v.
        x, y, z = [v.squeeze() for v in np.split(self._coords(mapindex), 3)]
        v = [self.probs[i][mapindex] for i in self.spatial_index[x, y, z]]
        return x, y, z, v

    def sample_locations(self, regionspec, numpoints, lower_threshold=0.0):
        """Sample 3D locations by using one of the maps as probability distributions.

        Parameters
        ----------
        regionspec: valid region specification
            Region to be used
        numpoints: int
            Number of samples to draw
        lower_threshold: float, default: 0
            Voxels in the map with a value smaller than this threshold will not be considered.

        Return
        ------
        samples : PointSet in physcial coordinates corresponding to this parcellationmap.

        TODO we can even circumvent fetch() and work with self._mapped_voxels to speed this up
        """
        if isinstance(regionspec, Number):
            mapindex = regionspec
        else:
            mapindex = self.get_index(regionspec)[0].map
        pmap = self.fetch(mapindex, cropped=True)
        D = np.array(pmap.dataobj)  # do a real copy so we don't modify the map
        D[D < lower_threshold] = 0.0
        p = (D / D.sum()).ravel()
        XYZ_ = np.array(
            np.unravel_index(np.random.choice(len(p), numpoints, p=p), D.shape)
        ).T
        XYZ = np.dot(pmap.affine, np.c_[XYZ_, np.ones(numpoints)].T)[:3, :].T
        return PointSet(XYZ, space=self.space)

    def fetch(
        self,
        mapindex: int = 0,
        resolution_mm: float = None,
        voi: BoundingBox = None,
        cropped=False,
    ):
        """
        Recreate a particular volumetric map from the sparse
        representation.

        Arguments
        ---------
        mapindex: int, or a valid region specification
            Index (or specification) of the map to be used
        resolution_mm: float
            Optional specification of a target resolution. Only used for neuroglancer volumes.
        voi: BoundingBox
            Optional specification of a bounding box
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

        if not isinstance(mapindex, Number):
            # assume we have some form of unique region specification
            logger.debug(
                f'Trying to decode map index for region specification "{mapindex}".'
            )
            mapindex = self.get_index(mapindex)[0].map

        x, y, z, v = self._mapped_voxels(mapindex)
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
                    for mapindex, value in self.probs[
                        self.spatial_index[x, y, z]
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


class LabelledSurface(ParcellationMap):
    """
    Represents a brain map in a surface space, with
    explicit knowledge about the region information per labelindex or channel.
    """

    def __init__(self, parcellation, space: Space):
        """
        Construct a labelled surface for the given parcellation and space.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        """
        assert space.type == "gii"
        super().__init__(parcellation, space, MapType.LABELLED)
        self.type = "gii-label"

    def _define_maps_and_regions(self):
        self._maploaders_cached = []
        self._regions_cached = {}

        with QUIET:
            tpl = self.space.get_template()
        for meshindex, meshname in enumerate(tpl.variants):

            labelsets = [
                v
                for v in self.parcellation.get_volumes(self.space)
                if v.volume_type == self.type and v.name == meshname
            ]
            assert len(labelsets) == 1
            labels = labelsets[0].fetch()
            unmatched = []
            for labelindex in np.unique(labels):
                if labelindex != 0:
                    pindex = ParcellationIndex(map=meshindex, label=labelindex)
                    try:
                        region = self.parcellation.decode_region(pindex)
                        if labelindex > 0:
                            self._regions_cached[pindex] = region
                    except ValueError:
                        unmatched.append(pindex)
            if unmatched:
                logger.warning(
                    f"{len(unmatched)} parcellation indices in labelled surface couldn't "
                    f"be matched to region definitions in {self.parcellation.name}"
                )

            self._maploaders_cached.append(
                lambda res=None, voi=None, variant=None, name=meshname, labels=labels: {
                    **self.space.get_template(variant=variant).fetch(name=name),
                    "labels": labels,
                }
            )

    def fetch_all(self, variant=None):
        """Get the combined mesh composed of all found submeshes (e.g. both hemispheres).

        Parameters
        -----------
        variant : str
            Optional specification of variant of the maps. For example,
            fsaverage provides the 'pial', 'white matter' and 'inflated' surface variants.
        """
        vertices = np.empty((0, 3))
        faces = np.empty((0, 3))
        labels = np.empty((0))
        for surfmap in self.fetch_iter(variant=variant):
            npoints = vertices.shape[0]
            vertices = np.append(vertices, surfmap["verts"], axis=0)
            faces = np.append(faces, surfmap["faces"] + npoints, axis=0)
            labels = np.append(labels, surfmap["labels"], axis=0)

        return dict(zip(["verts", "faces", "labels"], [vertices, faces, labels]))

    def colorize(self, values: dict, name: str = None, variant: str = None):
        """Colorize the parcellation mesh with the provided regional values.

        Parameters
        ----------
        values : dict
            Dictionary mapping regions to values
        name : str
            If specified, only submeshes matching this name are included, otherwise all meshes included.
        variant : str
            Optional specification of a specific variant to use for the maps. For example,
            fsaverage provides the 'pial', 'white matter' and 'inflated' surface variants.

        Return
        ------
        List of recolored parcellation meshes, each represented as a dictionary
        with elements
        - 'verts': An Nx3 array of vertex coordinates,
        - 'faces': an Mx3 array of face definitions using row indices of the vertex array
        - 'name': Name of the of the mesh variant
        NOTE: If a specific name was requested, the single mesh is returned instead of a list.
        """

        result = []
        for mapindex, mesh in enumerate(self.fetch_iter(variant=variant)):
            if (name is not None) and (name != mesh['name']):
                continue
            cmesh = {
                'verts': mesh['verts'],
                'faces': mesh['faces'],
                'labels': np.zeros_like(mesh['labels']),
                'name': mesh['name'],
            }
            for region, value in values.items():
                try:
                    indices = self.get_index(region)
                except IndexError:
                    continue
                for index in indices:
                    if index.map == mapindex:
                        cmesh['labels'][mesh['labels'] == index.label] = value
            result.append(cmesh)

        if len(result) == 1:
            return result[0]
        else:
            return result
