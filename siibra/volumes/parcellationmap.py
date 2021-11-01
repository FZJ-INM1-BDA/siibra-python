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

import numpy as np
from nibabel import Nifti1Image, funcs
from nilearn import image
from memoization import cached
from tqdm import tqdm
from abc import abstractmethod
from typing import Union

# Which types of available volumes should be preferred if multiple choices are available?
PREFERRED_VOLUMETYPES = ["nii", "neuroglancer/precomputed", "detailed maps"]


def create_map(parcellation, space: Space, maptype: MapType):
    """
    Creates a new ParcellationMap object of the given type.
    """
    classes = {
        MapType.LABELLED: LabelledParcellationMap,
        MapType.CONTINUOUS: ContinuousParcellationMap,
    }
    if maptype in classes:
        obj = classes[maptype](parcellation, space)
    elif maptype is None:
        logger.warning(
            "No maptype provided when requesting the parcellation map. Falling back to MapType.LABELLED"
        )
        obj = classes[MapType.LABELLED](parcellation, space)
    else:
        raise ValueError(f"Invalid maptype: '{maptype}'")
    if len(obj) == 0:
        raise ValueError(
            f"No data found to construct a {maptype} map for {parcellation.name} in {space.name}."
        )

    return obj


class ParcellationMap(ImageProvider):
    """
    Represents a brain map in a particular reference space, with
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

        self.maptype = maptype
        self.parcellation = parcellation
        self.space = space

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

    def fetch_iter(self, resolution_mm=None, voi: BoundingBox = None):
        """
        Returns an iterator to fetch all available maps sequentially.

        Parameters
        ----------
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes.
            If None, the smallest possible resolution will be chosen.
            If -1, the largest feasible resolution will be chosen.
        """
        logger.debug(f"Iterator for fetching {len(self)} parcellation maps")
        return (fnc(res=resolution_mm, voi=voi) for fnc in self.maploaders)

    def fetch(
        self, mapindex: int = 0, resolution_mm: float = None, voi: BoundingBox = None
    ):
        """
        Fetches the actual image data

        Parameters
        ----------
        mapindex : int
            The index of the available maps to be fetched.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes.
            If None, the smallest possible resolution will be chosen.
            If -1, the largest feasible resolution will be chosen.
        """
        if mapindex < len(self):
            if len(self) > 1:
                logger.info(
                    f"Returning map {mapindex+1} of in total {len(self)} available maps."
                )
            return self.maploaders[mapindex](res=resolution_mm, voi=voi)
        else:
            raise ValueError(
                f"'{len(self)}' maps available, but a mapindex of {mapindex} was requested."
            )

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

        for mapindex, img in tqdm(enumerate(self.fetch_iter()), total=N):
            out_data[mapindex] = np.asanyarray(img.dataobj)

        return funcs.squeeze_image(
            Nifti1Image(
                np.rollaxis(out_data, 0, out_data.ndim), im0.affine
            )
        )
        
    def fetch_regionmap(
        self,
        regionspec: Union[str, int, Region],
        resolution_mm=None,
        voi: BoundingBox = None,
    ):
        """
        Extract the mask for one particular region. For parcellation maps, this
        is a binary mask volume. For overlapping maps, this is the
        corresponding slice, which typically is a volume of float type.

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
        indices = self.decode_region(regionspec)
        mapimgs = []
        for index in indices:
            mapimg = self.fetch(
                resolution_mm=resolution_mm, mapindex=index.map, voi=voi
            )
            if index.label is not None:
                mapimg = Nifti1Image(
                    dataobj=(mapimg.get_fdata() == index.label).astype(np.uint8),
                    affine=mapimg.affine,
                )
            mapimgs.append(mapimg)

        if len(mapimgs) == 1:
            return mapimgs[0]
        elif self.maptype == MapType.LABELLED:
            m = mapimgs[0]
            for m2 in mapimgs[1:]:
                m.dataobj[m2.dataobj > 0] = 1
            return m
        else:
            logger.info(
                f"4D volume with {len(mapimgs)} continuous region maps extracted from region specification '{regionspec}'"
            )
            return image.concat_imgs(mapimgs)

    def get_shape(self, resolution_mm=None):
        return list(self.space.get_template().get_shape()) + [len(self)]

    def is_float(self):
        return self.maptype == MapType.CONTINUOUS

    def _load_regional_map(
        self, region: Region, resolution_mm, voi: BoundingBox = None, clip: bool = False
    ):
        logger.debug(f"Loading regional map for {region.name} in {self.space.name}")
        with QUIET:
            rmap = region.get_regional_map(self.space, self.maptype).fetch(
                resolution_mm=resolution_mm, voi=voi, clip=clip
            )
        return rmap

    @abstractmethod
    def assign_coordinates(self, xyz_phys, sigma_mm=1, sigma_truncation=3):
        """
        Implemented by derived classes.
        Assign regions to a physical coordinates with optional standard deviation.

        Parameters
        ----------
        xyz_phys : 3D point(s) in physical coordinates of the template space of the ParcellationMap
            Can be one 3D coordinate tuple, list of 3D tuples, Nx3 or Nx4 array of coordinate tuples,
            str of the form "3.1mm, -3.1mm, 80978mm", or list of such strings.
            See arrays.create_homogeneous_array
        sigma_mm : float (default: 1), applies only to continuous maps
            standard deviation /expected localization accuracy of the point, in
            mm units. For continuous maps, a 3D Gaussian distribution with that
            bandwidth will be used for representing the location.
        sigma_truncation : float (default: 3), applies only to continuous maps
            If sigma_phys is nonzero, this factor is used to determine where to
            truncate the Gaussian kernel in standard error units.
        """
        pass

    def __len__(self):
        """
        Returns the number of maps available in this parcellation.
        """
        return len(self.maploaders)

    def __contains__(self, spec):
        """
        Test if a 3D map identified by the given specification is included in this parcellation map.
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

    def decode_label(self, mapindex=None, labelindex=None):
        """
        Decode the region associated to a particular index.

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

    def decode_region(self, regionspec: Union[str, Region]):
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
        children_requested = set(region.leaves)
        if children_found != children_requested:
            raise IndexError(
                f"Cannot decode {regionspec} for the map in {self.space.name}, as it seems only partially mapped there."
            )
        return [idx for idx, _ in subregions]

class LabelledParcellationMap(ParcellationMap):
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

        # determine the map loader functions for each available map
        for volumetype in PREFERRED_VOLUMETYPES:
            sources = []
            for vsrc in self.parcellation.get_volumes(self.space.id):
                if vsrc.__class__.volume_type == volumetype:
                    sources.append(vsrc)
            if len(sources) > 0:
                break
        else:
            # reached only if for loop was not interrupted by 'break'
            raise RuntimeError(
                f"No suitable volume source for {self.parcellation.name} in {self.space.name}"
            )

        for source in sources:

            # Choose map loader function and populate label-to-region maps
            if source.volume_type == "detailed maps":
                self._maploaders_cached.append(
                    lambda res=None, voi=None: self._collect_maps(
                        resolution_mm=res, voi=voi
                    )
                )
                # collect all available region maps to maps label indices to regions
                current_index = 1
                for region in self.parcellation.regiontree:
                    with QUIET:
                        regionmap = region.get_regional_map(
                            self.space, MapType.LABELLED
                        )
                    if regionmap is not None:
                        self._regions_cached[
                            ParcellationIndex(map=0, label=current_index)
                        ] = region
                        current_index += 1

            elif source.volume_type == self.space.type:
                self._maploaders_cached.append(
                    lambda res=None, s=source, voi=None: self._load_map(
                        s, resolution_mm=res, voi=voi
                    )
                )
                # load map at lowest resolution to map label indices to regions
                mapindex = len(self._maploaders_cached) - 1
                with QUIET:
                    m = self._maploaders_cached[mapindex](res=None)
                unmatched = []
                for labelindex in np.unique(m.get_fdata()):
                    if labelindex != 0:
                        pindex = ParcellationIndex(map=mapindex, label=labelindex)
                        try:
                            region = self.parcellation.decode_region(pindex)
                            if labelindex > 0:
                                self._regions_cached[pindex] = region
                        except ValueError:
                            unmatched.append(pindex)
                if unmatched:
                    logger.warning(
                        f"{len(unmatched)} parcellation indices in labelled volume couldn't be matched to region definitions in {self.parcellation.name}"
                    )

    @cached
    def _load_map(self, volume: VolumeSrc, resolution_mm: float, voi: BoundingBox):
        m = volume.fetch(resolution_mm=resolution_mm, voi=voi)
        if len(m.dataobj.shape) == 4 and m.dataobj.shape[3] > 1:
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

        msg = f"Collecting {len(regions)} regional maps for '{self.space.name}'"
        current_index = 1
        for region in tqdm(regions, total=len(regions), desc=msg, unit="maps"):

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
            m.dataobj[mask.dataobj > 0] = current_index
            self._regions_cached[
                ParcellationIndex(map=0, label=current_index)
            ] = region
            current_index += 1

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
            indices = self.decode_region(region)
            for index in indices:
                if index.map not in maps:
                    # load the map
                    maps[index.map] = self.fetch(index.map)
                thismap = maps[index.map]
                if result is None:
                    # create the empty output
                    result = np.zeros_like(thismap.get_fdata())
                    affine = thismap.affine
                result[thismap.get_fdata()==index.label] = value
                
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
            enumerate(self.maploaders), total=len(self), desc=msg, unit=" maps"
        ):
            lmap = loadfnc()
            p2v = np.linalg.inv(lmap.affine)
            A = lmap.get_fdata()
            for i, coord in enumerate(coords):
                x, y, z = (np.dot(p2v, coord) + 0.5).astype("int")[:3]
                label = A[x, y, z]
                if label > 0:
                    region = self.decode_label(mapindex=mapindex, labelindex=label)
                    assignments[i].append((region, lmap, None))

        return assignments

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
            return tqdm(f, total=len(self.regions), desc=msg, unit="regions")

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
            (region, pmaps[region], scores)
            for region, scores in sorted(
                values.items(),
                key=lambda item: abs(item[1]["correlation"]),
                reverse=True,
            )
        ]
        return assignments


class ContinuousParcellationMap(ParcellationMap):
    """
    Represents a brain map in a particular reference space, with
    explicit knowledge about the region information per labelindex or channel.

    This form represents overlapping regional maps (often probability maps) (MapType.CONTINUOUS)
    where each "time"-slice is a 3D volume representing
    a map of a particular brain region. This format is used for
    probability maps and similar continuous forms. The number of
    regions correspond to the z dimension of the 4 object.
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
        super().__init__(parcellation, space, MapType.CONTINUOUS)

    def _define_maps_and_regions(self):
        self._maploaders_cached = []
        self._regions_cached = {}

        # Multiple volume sources could be given - find the preferred one
        volume_sources = sorted(
            self.parcellation.get_volumes(self.space.id),
            key=lambda vsrc: PREFERRED_VOLUMETYPES.index(vsrc.volume_type),
        )

        for source in volume_sources:

            if (
                isinstance(source, ImageProvider)
                and source.is_float()
                and source.is_4D()
                and source.get_shape()[3] > 1
            ):

                # The source is 4D float, that's what we are looking for.
                # We assume the fourth dimension contains the regional continuous maps.
                nmaps = source.get_shape()[3]
                logger.info(
                    f"{nmaps} continuous maps will be extracted from 4D volume for {self.parcellation}."
                )
                for i in range(nmaps):
                    self._maploaders_cached.append(
                        lambda res=None, voi=None, mi=i: source.fetch(
                            resolution_mm=res, voi=voi, mapindex=mi
                        )
                    )
                    region = self.parcellation.decode_region(i + 1)
                    pindex = ParcellationIndex(map=i, label=None)
                    self._regions_cached[pindex] = region

                # we are finished, no need to look for regional map.
                return

        # otherwise we look for continuous maps associated to individual regions
        i = 0
        for region in self.parcellation.regiontree:
            with QUIET:
                regionmap = region.get_regional_map(self.space, MapType.CONTINUOUS)
            if regionmap is None:
                continue
            if region in self.regions.values():
                logger.debug(f"Region already seen in tree: {region.key}")
                continue
            self._maploaders_cached.append(
                lambda r=region, res=None, voi=None: self._load_regional_map(
                    r, resolution_mm=res, voi=voi
                )
            )
            pindex = ParcellationIndex(map=i, label=None)
            self._regions_cached[pindex] = region
            i += 1
        logger.info(
            f"{i} regional continuous maps found for {self.parcellation} in {self.space.name}."
        )

    @cached
    def assign_coordinates(
        self, point: Union[Point, PointSet], sigma_mm=1, sigma_truncation=3
    ):
        """
        Assign regions to a physical coordinates with optional standard deviation.

        Parameters
        ----------
        point : Point or PointSet
        sigma_mm : float (default: 1)
            standard deviation /expected localization accuracy of the point, in
            mm units. A 3D Gaussian distribution with that
            bandwidth will be used for representing the location.
        sigma_truncation : float (default: 3)
            If sigma_phys is nonzero, this factor is used to determine where to
            truncate the Gaussian kernel in standard error units.
        """
        assert sigma_mm >= 1

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

        # convert sigma to voxel coordinates
        tpl = self.space.get_template().fetch()
        phys2vox = np.linalg.inv(tpl.affine)
        scaling = np.array([np.linalg.norm(tpl.affine[:, i]) for i in range(3)]).mean()
        sigma_vox = sigma_mm / scaling

        if sigma_vox < 3:
            N = len(self)
            msg = f"Assigning {len(coords)} coordinates to {N} maps"
            assignments = [[] for _ in coords]
            for mapindex, loadfnc in tqdm(
                enumerate(self.maploaders), total=len(self), desc=msg, unit=" maps"
            ):
                pmap = loadfnc()
                p2v = np.linalg.inv(tpl.affine)
                A = pmap.get_fdata()
                region = self.decode_label(mapindex=mapindex)
                for i, coord in enumerate(coords):
                    x, y, z = (np.dot(p2v, coord) + 0.5).astype("int")[:3]
                    value = A[x, y, z]
                    if value > 0:
                        assignments[i].append((region, pmap, value))
        else:
            logger.info(
                (
                    f"Assigning {len(coords)} uncertain coordinates (stderr={sigma_mm}) to {len(self)} maps."
                )
            )
            kernel = create_gaussian_kernel(sigma_vox, sigma_truncation)
            r = int(kernel.shape[0] / 2)  # effective radius
            assignments = []
            for coord in coords:
                xyz_vox = (np.dot(phys2vox, coord) + 0.5).astype("int")
                shift = np.identity(4)
                shift[:3, -1] = xyz_vox[:3] - r
                W = Nifti1Image(dataobj=kernel, affine=np.dot(tpl.affine, shift))
                assignments.append(
                    self.assign(W, msg=", ".join([f"{v:.1f}" for v in coord[:3]]))
                )

        if len(assignments) == 1:
            return assignments[0]
        else:
            return assignments

    def assign(self, img: Nifti1Image, msg=None, quiet=False):
        """
        Assign the region of interest represented by a given volumetric image to continuous brain regions in this map.

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
            msg = f"Assigning structure to {len(self)} maps"

        # How to visualize progress from the iterator?
        def plain_progress(f):
            return f

        def visual_progress(f):
            return tqdm(f, total=len(self), desc=msg, unit="maps")

        progress = plain_progress if quiet else visual_progress

        # setup assignment loop
        values = {}
        pmaps = {}

        for mapindex, loadfnc in progress(enumerate(self.maploaders)):

            # load the regional map
            this = loadfnc()
            if not this:
                logger.warning(
                    f"Could not load regional map for {self.regions[mapindex].name}"
                )
                continue

            scores = compare_maps(img, this)
            if scores["overlap"] > 0:
                pmaps[mapindex] = this
                values[mapindex] = scores

        assignments = [
            (self.decode_label(mapindex=i), pmaps[i], scores)
            for i, scores in sorted(
                values.items(),
                key=lambda item: abs(item[1]["correlation"]),
                reverse=True,
            )
        ]
        return assignments

    def colorize(self, values: dict):
            """Produce a colorized 3D map by projecting the regional values to regional maps.

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
                indices = self.decode_region(region)
                for index in indices:
                    if index.map not in maps:
                        # load the map
                        with QUIET:
                            maps[index.map] = self.fetch(mapindex=index.map)
                    thismap = maps[index.map]
                    if result is None:
                        # create the empty output
                        result = np.zeros_like(thismap.get_fdata())
                        affine = thismap.affine
                    result = np.maximum(result, thismap.get_fdata()*value)
                    
            return Nifti1Image(result, affine) 