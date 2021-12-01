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
from ..retrieval import CACHE

import numpy as np
import nibabel as nib
from nilearn import image
from memoization import cached
from tqdm import tqdm
from abc import abstractmethod
from typing import Union
from os import path

# Which types of available volumes should be preferred if multiple choices are available?
PREFERRED_VOLUMETYPES = ["nii", "neuroglancer/precomputed", "detailed maps"]


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

    _instances = {}

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
        self._regions_cached = {}

    @classmethod
    def get_instance(cls, parcellation, space: Space, maptype: MapType):
        """
        Returns the ParcellationMap object of the requested type.
        """
        key = (parcellation.key, space.key, maptype)

        if key in cls._instances:
            # Instance already available - do not build another object
            return cls._instances[key]

        # create a new object
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

        cls._instances[key] = obj
        return obj

    @property
    def regions(self):
        """
        Dictionary of regions associated to the parcellion map, indexed by ParcellationIndex.
        Lazy implementation - self._link_regions() will be called when the regions are accessed for the first time.
        """
        return self._regions_cached

    @property
    def names(self):
        return self.parcellation.names

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
        #return (fnc(res=resolution_mm, voi=voi) for fnc in self.maploaders)
        return (self.fetch(mapindex, resolution_mm, voi) for mapindex in range(len(self)))

    @abstractmethod
    def fetch(
        self, mapindex: int = 0, resolution_mm: float = None, voi: BoundingBox = None
    ):
        """
        Fetches one particular map. Implemented in derived classes.

        Parameters
        ----------
        mapindex : int
            The index of the available maps to be fetched.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes.
            If None, the smallest possible resolution will be chosen.
            If -1, the largest feasible resolution will be chosen.
        """
        pass

    @abstractmethod
    def __len__(self):
        """ 
        The number of 3D maps provided by this parcellation map. 
        Implemented by derived classes. 
        """
        pass

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

        return nib.funcs.squeeze_image(
            nib.Nifti1Image(
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
        indices = self.decode_region(regionspec)
        data = None
        affine = None
        for index in indices:
            with QUIET:
                mapimg = self.fetch(
                    resolution_mm=resolution_mm, mapindex=index.map, voi=voi
                )
            if index.label is None: # region is defined by the whole map
                newdata = mapimg.get_fdata()
            else: # region is defined by a particular label
                newdata = (mapimg.get_fdata() == index.label).astype(np.uint8)
            if data is None:
                data = newdata
                affine = mapimg.affine
            else:
                data = np.maximum(data, newdata)

        return nib.Nifti1Image(data, affine)

    def _load_regional_map(
        self, region: Region, resolution_mm, voi: BoundingBox = None, clip: bool = False
    ):
        logger.debug(f"Loading regional map for {region.name} in {self.space.name}")
        with QUIET:
            rmap = region.get_regional_map(self.space, self.maptype).fetch(
                resolution_mm=resolution_mm, voi=voi, clip=clip
            )
        return rmap

    def get_shape(self, resolution_mm=None):
        return list(self.space.get_template().get_shape()) + [len(self)]

    def is_float(self):
        return self.maptype == MapType.CONTINUOUS

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

    _regions_cached = None
    _maploaders_cached = None

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

    def __len__(self):
        """
        Returns the number of maps available in this parcellation.
        """
        return len(self.maploaders)

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
            m = nib.Nifti1Image(dataobj=np.asarray(m.dataobj, dtype=int), affine=m.affine)
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
                m = nib.Nifti1Image(
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
                
        return nib.Nifti1Image(result, affine)    

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

    def assign(self, img: nib.Nifti1Image, msg=None, quiet=False):
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
            (region, region.index.map, scores)
            for region, scores in sorted(
                values.items(),
                key=lambda item: abs(item[1]["correlation"]),
                reverse=True,
            )
        ]
        return assignments


class ContinuousParcellationMap(ParcellationMap):
    """ A sparsely representation of list of continuous (e.g. probabilistic) brain region maps. 
    
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

    def __init__(self, parcellation, space):

        ParcellationMap.__init__(self, parcellation, space, maptype="continuous")

        # Check for available maps and brain regions.
        # First look for a 4D array where the last dimension are the different maps
        self._maploaders = []
        self._map4d = None
        for v in self.parcellation.volumes:
            if isinstance(v, ImageProvider) and v.is_float() and v.is_4D() and v.get_shape()[3] > 1:
                self._map4d = v.fetch()
                for mapindex in range(self._map4d.shape[3]):
                    self._maploaders.append(lambda m=mapindex: self._map4d.slicer[:,:,:,m])
                    # TODO this might not be correct for parcellations other than DifumoXX
                    r = self.parcellation.decode_region(mapindex + 1)
                    self._regions_cached[ParcellationIndex(map=mapindex, label=None)] = r

        if self._map4d is None:
            # No 4D array, look for regional continuous maps stored in the region tree.
            mapindex = 0
            for r in self.parcellation.regiontree.leaves:
                if r in self.regions.values():
                    continue
                if r.has_regional_map(self.space, self.maptype):
                    regionmap = r.get_regional_map(self.space, self.maptype)
                    self._maploaders.append(lambda r=regionmap: r.fetch())
                    self._regions_cached[ParcellationIndex(map=mapindex, label=None)] = r
                    mapindex += 1

        # either load or build the sparse index
        if not self._load_index():
            self._build_index()
            self._store_index()
        assert self.spatial_index.max() == len(self.probs)-1

        logger.info(f"Constructed {self.__class__.__name__} for {self.parcellation.name} with {len(self)} maps.")

    def _load_index(self):

        self.spatial_index = None
        self.probs = []
        self.bboxes = []
        self.affine = None

        prefix = f"{self.parcellation.id}_{self.space.id}_{self.maptype}_index"
        probsfile = CACHE.build_filename(f"{prefix}", suffix = 'probs.txt')
        bboxfile = CACHE.build_filename(f"{prefix}", suffix = 'bboxes.txt')
        indexfile = CACHE.build_filename(f"{prefix}", suffix = 'index.nii.gz')
        
        if not all(path.isfile(f) for f in [probsfile, bboxfile, indexfile]):
            return False

        logger.info(f"Loading continuous map index for {len(self)} brain regions.")
        indeximg = nib.load(indexfile)
        self.spatial_index = np.asanyarray(indeximg.dataobj)
        self.affine = indeximg.affine

        with open(probsfile, 'r') as f:
            lines = f.readlines()
            msg = f"Loading spatial index for location assignment"
            for line in tqdm(lines, total=len(lines), desc=msg, unit="voxels"):
                fields = line.split(' ')
                mapindices = list(map(int, fields[0::2]))
                values = list(map(float, fields[1::2]))
                D = dict(zip(mapindices, values))
                self.probs.append(D)
        
        with open(bboxfile, 'r') as f:
            for line in f:
                fields = line.split(' ')
                self.bboxes.append({
                    "minpoint": tuple(map(int, fields[:3])),
                    "maxpoint": tuple(map(int, fields[3:]))
                })

        return True

    def _store_index(self):
        # store spatial index and probability list to file
        prefix = f"{self.parcellation.id}_{self.space.id}_{self.maptype}_index"
        probsfile = CACHE.build_filename(f"{prefix}", suffix = 'probs.txt')
        bboxfile = CACHE.build_filename(f"{prefix}", suffix = 'bboxes.txt')
        indexfile = CACHE.build_filename(f"{prefix}", suffix = 'index.nii.gz')

        nib.Nifti1Image(self.spatial_index, self.affine).to_filename(indexfile)

        with open(probsfile, 'w') as f:
            for D in self.probs:
                f.write("{}\n".format(
                        ' '.join(f'{i} {p}' for i, p in D.items())
                ))

        with open(bboxfile, 'w') as f:
            for bbox in self.bboxes:
                f.write("{} {}\n".format(
                    ' '.join(map(str, bbox["minpoint"])),
                    ' '.join(map(str, bbox["maxpoint"]))
                ))

    def _build_index(self):
        """ Load map image with the given index. """

        logger.info(
            f"Creating the spatial index for {self.parcellation.name} continuous maps "
            f"in {self.space.name}. This will take a minute, but is only performed once."
        )

        self.probs = []
        self.bboxes = []
        self.spatial_index = None
        self.affine = None
        for mapindex in tqdm(
            range(len(self)), total=len(self), unit="maps", 
            desc=f"Fetching {len(self)} volumetric maps"
        ):
            with QUIET:
                # retrieve the probability map
                img = self._maploaders[mapindex]()

            if self.spatial_index is None:
                self.spatial_index = np.zeros(img.shape, dtype=np.int32)-1
                self.affine = img.affine
            else:
                assert img.shape == self.shape
                assert (img.affine - self.affine).sum() == 0

            imgdata = np.asanyarray(img.dataobj)
            X, Y, Z = [v.astype('int32') for v in np.where(imgdata > 0)]
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
            
            self.bboxes.append({
                'minpoint' : (X.min(), Y.min(), Z.min()),
                'maxpoint' : (X.max(), Y.max(), Z.max())
            })

    def __len__(self):
        return len(self._maploaders)

    @property
    def shape(self):
        return self.spatial_index.shape

    def _coords(self, mapindex):
        # Nx3 array with x/y/z coordinates of the N nonzero values of the given mapindex
        coord_ids = [i for i, l in enumerate(self.probs) if mapindex in l]
        x0, y0, z0 = self.bboxes[mapindex]['minpoint']
        x1, y1, z1 = self.bboxes[mapindex]['maxpoint']
        return (np.array(
            np.where(
                np.isin(self.spatial_index[x0:x1+1, y0:y1+1, z0:z1+1], coord_ids)
            )
        ).T + (x0, y0, z0)).T

    def _mapped_voxels(self, mapindex):
        # returns the x, y, and z coordinates of nonzero voxels for the map 
        # with the given index, together with their corresponding values v. 
        x, y, z = [v.squeeze() for v in np.split(self._coords(mapindex), 3)]
        v = [self.probs[i][mapindex] for i in self.spatial_index[x, y, z]]
        return x, y, z, v

    def fetch(self, mapindex: int = 0, resolution_mm: float = None, voi: BoundingBox = None, cropped=False):
        """ 
        Recreate a particular volumetric map from the sparse
        representation. 
        """
        if voi is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support volume of interest fetching yet."
            )
        if resolution_mm is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support fetching at resolutions other than 1mm yet."
            )

        x, y, z, v = self._mapped_voxels(mapindex)
        if cropped:
            bbox = np.array([[min(_), max(_)] for _ in [x,y,z]])
            result = np.zeros(bbox[:,1]-bbox[:,0]+1)
            x0, y0, z0 = bbox[:,0]
            result[x-x0, y-y0, z-z0] = v
            shift = np.identity(4)
            shift[:3,-1] = bbox[:,0]            
            return nib.Nifti1Image(result, np.dot(self.affine, shift))
        else:
            result = np.zeros(self.shape, dtype=np.float32)
            result[x, y, z] = v
            return nib.Nifti1Image(result, self.affine)
 
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
        scaling = np.array([np.linalg.norm(self.affine[:, i]) for i in range(3)]).mean()
        sigma_vox = sigma_mm / scaling
        phys2vox = np.linalg.inv(self.affine)

        if sigma_vox < 3:
            N = len(self)
            logger.info(f"Assigning {len(coords)} coordinates to {N} maps")
            assignments = [[] for _ in coords]
            for i, coord in enumerate(coords):
                x, y, z = (np.dot(phys2vox, coord) + 0.5).astype("int")[:3]
                for mapindex, value in self.probs[self.spatial_index[x,y,z]].items():
                    region = self.decode_label(mapindex=mapindex)
                    if value > 0:
                        assignments[i].append((region, mapindex, value))
                assignments[i] = sorted(assignments[i], key=lambda d: d[2])
        else:
            logger.info(
                f"Assigning {len(coords)} uncertain coordinates "
                f"(stderr={sigma_mm}) to {len(self)} maps."
            )
            kernel = create_gaussian_kernel(sigma_vox, sigma_truncation)
            r = int(kernel.shape[0] / 2)  # effective radius
            assignments = []
            for coord in coords:
                xyz_vox = (np.dot(phys2vox, coord) + 0.5).astype("int")
                shift = np.identity(4)
                shift[:3, -1] = xyz_vox[:3] - r
                W = nib.Nifti1Image(dataobj=kernel, affine=np.dot(self.affine, shift))
                assignments.append(
                    self.assign(W, msg=", ".join([f"{v:.1f}" for v in coord[:3]]))
                )

        if len(assignments) == 1:
            return assignments[0]
        else:
            return assignments
    

    def assign(self, img: nib.Nifti1Image, msg=None, quiet=False):
        
        # ensure query image is in parcellation map's voxel space
        if (img.affine-self.affine).sum() == 0:
            img2 = img
        else:
            img2 = image.resample_img( img,
                target_affine = self.affine,
                target_shape = self.shape   
            )

        bbox2 = BoundingBox.from_image(img2, None, ignore_affine=True)
        assignments = []

        for mapindex in tqdm(range(len(self)), total=len(self), unit=" map", desc=msg):
            
            bbox1 = BoundingBox(
                self.bboxes[mapindex]['minpoint'], 
                self.bboxes[mapindex]['maxpoint'], 
                space=None)
            if bbox1.intersection(bbox2) is None:
                continue
            
            # compute union of voxel space bounding boxes
            bbox = bbox1.union(bbox2)
            bbshape = np.array(bbox.shape, dtype='int')+1
            x0, y0, z0 = map(int, bbox.minpoint)
            x1, y1, z1 = [int(v)+1 for v in bbox.maxpoint] 
            
            # build flattened vector of map values
            v1 = np.zeros(np.prod(bbshape))
            XYZ = self._coords(mapindex).T
            x, y, z = [v.squeeze() for v in np.split(XYZ, 3, axis=1)]
            indices = np.ravel_multi_index((x-x0, y-y0, z-z0), bbshape)
            v1_ = [self.probs[i][mapindex] for i in self.spatial_index[x, y, z]]
            v1[indices] = v1_

            # build flattened vector of query image values
            v2 = img2.dataobj[x0:x1, y0:y1, z0:z1].ravel()
            assert v1.shape == v2.shape
            
            intersection = np.minimum(v1, v2).sum()
            if intersection == 0:
                continue

            v1d = v1 - v1.mean()
            v2d = v2 - v2.mean()
            rho = (v1d * v2d).sum() / np.sqrt((v1d**2).sum()) / np.sqrt((v2d**2).sum())
                    
            scores = {
                "overlap": intersection / np.maximum(v1, v2).sum(),
                "contained": intersection / v1.sum(),
                "contains": intersection / v2.sum(),
                "correlation": rho
            }
            region = self.decode_label(mapindex=mapindex, labelindex=None)
            assignments.append((region, mapindex, scores))

        return sorted(assignments, key=lambda d: -abs(d[2]['correlation']))

