# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy.core.fromnumeric import argmax
from . import logger
from .space import Space,SpaceVOI
from .commons import MapType
from .volumesrc import ImageProvider
from .arrays import create_homogeneous_array,create_gaussian_kernel,argmax_dim4
from .region import Region

import numpy as np
import nibabel as nib
from nilearn import image
from memoization import cached
from tqdm import tqdm
from abc import abstractmethod

# Which types of available volumes should be preferred if multiple choices are available?
PREFERRED_VOLUMETYPES = ['nii','neuroglancer/precomputed','detailed maps']

def create_map(parcellation, space:Space, maptype:MapType ):
    """
    Creates a new ParcellationMap object of the given type.
    """
    classes = {
        MapType.LABELLED:LabelledParcellationMap,
        MapType.CONTINUOUS:ContinuousParcellationMap
    }
    if maptype in classes:
        return classes[maptype](parcellation,space)
    elif maptype is None:
        logger.warning('No maptype provided when requesting the parcellation map. Falling back to maptype LABELLED')
        return classes[MapType.LABELLED](parcellation,space)
    else:
        raise ValueError(f"Invalid maptype: '{maptype}'")

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

    def __init__(self, parcellation, space: Space, maptype=MapType ):
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
            raise ValueError( 'Parcellation "{}" does not provide a map for space "{}"'.format(
                parcellation.name, space.name ))

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
        Dictionary of regions associated to the parcellion map, indexed by (labelindex,mapindex).
        Lazy implementation - self._link_regions() will be called when the regions are accessed for the first time.
        """
        if self._regions_cached is None:
            self._define_maps_and_regions()
        return self._regions_cached

    @abstractmethod
    def _define_maps_and_regions(self):
        pass

    def fetchall(self,resolution_mm=None,voi:SpaceVOI=None):
        """
        Returns an iterator to fetch all available maps sequentially:

        Parameters
        ----------
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes. 
            If None, the smallest possible resolution will be chosen. 
            If -1, the largest feasible resolution will be chosen.        
        """
        logger.debug(f'Iterator for fetching {len(self)} parcellation maps')
        return (fnc(res=resolution_mm,voi=voi) for fnc in self.maploaders)

    def fetch(self,resolution_mm=None,mapindex=0, voi:SpaceVOI=None):
        """
        Fetches the actual image data

        Parameters
        ----------
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes. 
            If None, the smallest possible resolution will be chosen. 
            If -1, the largest feasible resolution will be chosen.  
        mapindex : int
            The index of the available maps to be fetched.       
        """
        if mapindex<len(self):
            return self.maploaders[mapindex](res=resolution_mm,voi=voi)
        else:
            raise ValueError(f"'{len(self)}' maps available, but a mapindex of {mapindex} was requested.")

    @cached
    def _load_regional_map(self, region:Region, resolution_mm, voi:SpaceVOI=None):
        logger.debug(f"Loading regional map for {region.name} in {self.space.name}")
        return region.get_regional_map(self.space, self.maptype).fetch(resolution_mm=resolution_mm,voi=voi)

    def __len__(self):
        """
        Returns the number of maps available in this parcellation.
        """
        return len(self.maploaders)

    def __contains__(self,spec):
        """
        Test if a 3D map identified by the given specification is included in this parcellation map. 
        For integer values, it is checked wether a corresponding slice along the fourth dimension could be extracted.
        Alternatively, a region object can be provided, and it will be checked wether the region is mapped.
        You might find the decode_region() function of Parcellation and Region objects useful for the latter.
        """
        if isinstance(spec,int):
            return spec in range(len(self.maploaders))
        elif isinstance(spec,Region):
            for _,region in self.regions.items():
                if region==spec:
                    return True
        return False

    @abstractmethod
    def decode_label(self,index:int,mapindex=None):
        """
        Decode the region associated to a particular index.
        for CONTINUOUS, this is the position of the continous map in the stack (ie. z index)
        For LABELLED types, this is the labelindex, ie. the color at a given voxel.
        For LABELLED types with multiple maps, the map index can be provided in addition.

        Parameters
        ----------
        index : int
            The index
        mapindex : int, or None (default=None)
            Index of the map, in a labelled volume with more than
            a single parcellation map.
        """
        pass

    @abstractmethod
    def extract_mask(self,region:Region, resolution_mm=None):
        """
        Extract the mask for one particular region. For parcellation maps, this
        is a binary mask volume. For overlapping maps, this is the
        corresponding slice, which typically is a volume of float type.

        Parameters
        ----------
        region : Region
            The desired region.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes. 
            If None, the smallest possible resolution will be chosen. 
            If -1, the largest feasible resolution will be chosen. 

        Return
        ------
        Nifti1Image, if found, otherwise None
        """
        pass
    
    def get_shape(self,resolution_mm=None):
        return list(self.space.get_template().get_shape()) + [len(self)]

    def is_float(self):
        return self.maptype==MapType.CONTINUOUS

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

    def __init__(self, parcellation, space: Space ):
        """
        Construct a ParcellationMap for the given parcellation and space.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        """
        super().__init__(parcellation, space,MapType.LABELLED)

    def _define_maps_and_regions(self):

        self._maploaders_cached=[]
        self._regions_cached = {}

        # determine the map loader functions for each available map
        for mapname in self.parcellation.volume_src[self.space]:

            # Determine the preferred volume source for loading the parcellation map
            volume_sources = sorted(
                    self.parcellation.volume_src[self.space][mapname],
                    key=lambda vsrc: PREFERRED_VOLUMETYPES.index(vsrc.volume_type))
            if len(volume_sources)==0:
                logger.error(f'No suitable volume source for {self.parcellation.name} in {self.space.name}')
                continue
            source = volume_sources[0]

            # Choose map loader function
            if source.volume_type=="detailed maps":
                self._maploaders_cached.append(lambda res=None,voi=None: self._collect_maps(resolution_mm=res,voi=voi))
            elif source.volume_type==self.space.type:
                self._maploaders_cached.append(lambda res=None,s=source,voi=None: self._load_map(s,resolution_mm=res,voi=voi))

            # load map at lowest resolution
            mapindex = len(self._maploaders_cached)-1
            m = self._maploaders_cached[mapindex](res=None)
            assert(m is not None)
            
            # map label indices to regions
            unmatched_labels = []
            for labelindex in np.unique(np.asarray(m.dataobj)):
                if labelindex!=0:
                    try:
                        region = self.parcellation.decode_region(int(labelindex),mapindex)
                        if labelindex>0:
                            self._regions_cached[labelindex,mapindex] = region
                    except ValueError:
                        unmatched_labels.append(labelindex)

            if unmatched_labels:
                logger.warning(f"{len(unmatched_labels)} labels in labelled volume couldn't be matched to region definitions in {self.parcellation.name}: {unmatched_labels}")
            
    @cached
    def _load_map(self,volume_src,resolution_mm,voi):
        m = volume_src.fetch(resolution_mm,voi=voi)
        if len(m.dataobj.shape)==4 and m.dataobj.shape[3]>1:
            logger.info(f"{m.dataobj.shape[3]} continuous maps given - using argmax to generate a labelled volume. ")
            m = argmax_dim4(m)
        if m.dataobj.dtype.kind=='f':
            raise RuntimeError("Floating point image type encountered when building a labelled volume for {self.parcellation.name}.")
        return m

    @cached
    def _collect_maps(self,resolution_mm,voi):
        """
        Build a 3D volume from the list of available regional maps.

        Return
        ------
        Nifti1Image, or None if no maps are found.
        
        """
        m = None

        # generate empty mask covering the template space
        tpl = self.space.get_template().fetch(resolution_mm,voi=voi)
        m = nib.Nifti1Image(np.zeros_like(tpl.dataobj,dtype='uint'),tpl.affine)

        # collect all available region maps
        regions = [r for r in self.parcellation.regiontree 
                if r.has_regional_map(self.space,MapType.LABELLED)]

        msg =f"Loading {len(regions)} regional maps for space '{self.space.name}'..."
        logger.info(msg)
        for region in tqdm(regions,total=len(regions)):
            assert(region.labelindex)
            # load region mask
            mask_ = self._load_regional_map(region,resolution_mm=resolution_mm,voi=voi)
            if not mask_:
                continue
            # build up the aggregated mask with labelled indices
            if mask_.shape!=m.shape:
                mask = image.resample_to_img(mask_,m,interpolation='nearest')
            else:
                mask = mask_
            m.dataobj[mask.dataobj>0] = region.labelindex

        return m

    def decode_label(self,index:int,mapindex=None):
        """
        Decode the region associated to a particular index.
        For LABELLED types, this is the labelindex, ie. the color at a given voxel.
        For LABELLED types with multiple maps, the map index can be provided in addition.

        Parameters
        ----------
        index : int
            The index
        mapindex : int, or None (default=None)
            Index of the map, in a labelled volume with more than
            a single parcellation map.
        """
        if mapindex is None:
            for ix,mi in self.regions.keys():
                if ix==index:
                    return self.regions[index,mi]
            raise ValueError(f"Could not decode label index {index} (mapindex {mapindex})")
        else:
            return self.regions[index,mapindex]


    def extract_mask(self,region:Region,resolution_mm=None):
        """
        Extract the binary mask for one particular region. 

        Parameters
        ----------
        region : Region
            The desired region.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes. 
            If None, the smallest possible resolution will be chosen. 
            If -1, the largest feasible resolution will be chosen. 

        Return
        ------
        Nifti1Image, if found, otherwise None
        """
        if not region in self:
            return None
        mapimg = self.fetch(resolution_mm=resolution_mm,mapindex=region.mapindex)
        index = region.labelindex
        return nib.Nifti1Image(
                dataobj=(np.asarray(mapimg.dataobj)==index).astype(int),
                affine=mapimg.affine)


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

    def __init__(self, parcellation, space: Space ):
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
        self._maploaders_cached=[]
        self._regions_cached={}

        # check for maps associated to the parcellations
        for mapname in self.parcellation.volume_src[self.space]:

            # Multiple volume sources could be given - find the preferred one
            volume_sources = sorted(
                    self.parcellation.volume_src[self.space][mapname],
                    key=lambda vsrc: PREFERRED_VOLUMETYPES.index(vsrc.volume_type))
            if len(volume_sources)==0:
                logger.error(f'No suitable volume source for "{mapname}"' +
                             f'of {self.parcellation.name} in {self.space.name}')
                continue
            source = volume_sources[0]
            
            if not all([source.is_float(),source.is_4D()]):
                continue
            if source.get_shape()[3]<2:
                continue

            #  The source is 4D float! We assume the fourth dimension contains the regional continuous maps.
            nmaps = source.get_shape()[3]
            logger.info(f'{nmaps} continuous maps will be extracted from 4D volume for {self.parcellation}.')
            for i in range(nmaps):
                self._maploaders_cached.append(
                    lambda res=None,voi=None,mi=i: source.fetch(resolution_mm=res,voi=voi,mapindex=mi))
                region = self.parcellation.decode_region(i+1)
                self._regions_cached[-1,i] = region

            # we are finished, no need to look for regional map.
            return
                
        # otherwise we look for continuous maps associated to individual regions
        regions = [r for r in self.parcellation.regiontree 
                if r.has_regional_map(self.space,MapType.CONTINUOUS)]
        logger.info(f'{len(regions)} regional continuous maps found for {self.parcellation}.')
        labelindex = -1
        for mapindex,region in enumerate(regions):
            self._maploaders_cached.append(
                lambda r=region,res=None,voi=None:self._load_regional_map(r,resolution_mm=res,voi=voi))
            if region in self.regions.values():
                logger.debug(f"Region already seen in tree: {region.key}")
            self._regions_cached[-1,mapindex] = region

    def decode_label(self,index:int,mapindex=None):
        """
        Decode the region associated to a particular index.
        for CONTINUOUS, this is the position of the continous map in the stack (ie. z index)

        Parameters
        ----------
        index : int
            The index
        mapindex : int, or None (default=None)
            Index of the map, in a labelled volume with more than
            a single parcellation map.
        """
        return self.regions[-1,index]

    def extract_mask(self,region:Region,resolution_mm=None):
        """
        Extract the mask for one particular region. For parcellation maps, this
        is a binary mask volume. For overlapping maps, this is the
        corresponding slice, which typically is a volume of float type.

        Parameters
        ----------
        region : Region
            The desired region.
        resolution_mm : float or None (optional)
            Physical resolution of the map, used for multi-resolution image volumes. 
            If None, the smallest possible resolution will be chosen. 
            If -1, the largest feasible resolution will be chosen. 

        Return
        ------
        Nifti1Image, if found, otherwise None
        """
        if not region in self:
            return None
        return self.fetch(resolution_mm=resolution_mm, mapindex=region.mapindex)

    @cached
    def assign_regions(self,xyz_phys,sigma_mm=0,sigma_truncation=3):
        """
        Assign regions to a physical coordinates with optional standard deviation.

        TODO  allow to process multiple xyz coordinates at once

        Parameters
        ----------
        xyz_phys : 3D coordinate tuple, list of 3D tuples, or Nx3 array of coordinate tuples
            3D point(s) in physical coordinates of the template space of the
            ParcellationMap
        sigma_mm : float (default: 0)
            standard deviation /expected localization accuracy of the point, in
            mm units. If nonzero, A 3D Gaussian distribution with that
            bandwidth will be used for representing the location instead of a
            deterministic coordinate.
        sigma_truncation : float (default: 3)
            If sigma_phys is nonzero, this factor is used to determine where to
            truncate the Gaussian kernel in standard error units.
        thres_percent : float (default: 1)
            Matching regions with a value of the continous map below this threshold will not be returned.
        print_report : Boolean (default: True)
            Wether to print a short report to stdout
        """

        # Convert input to Nx4 list of homogenous coordinates
        assert(len(xyz_phys)>0)
        XYZH = create_homogeneous_array(xyz_phys)
        numpts = XYZH.shape[0]

        if sigma_mm>0:
            logger.info((f"Assigning {numpts} uncertain coordinates (stderr={sigma_mm}) to {len(self)} maps." ))
        else:
            logger.info((f"Assigning {numpts} deterministic coordinates to {len(self)} maps."))

        values = {i:[] for i in range(numpts)}
        for mapindex,loadfnc in tqdm(enumerate(self.maploaders),total=len(self)):

            pmap = loadfnc()
            assert(pmap.dataobj.dtype.kind=='f')
            if not pmap:
                logger.warning(f"Could not load regional map for {self.regions[-1,mapindex].name}")
                for i in range(numpts):
                    values[i].append(-1)
                continue
            phys2vox = np.linalg.inv(pmap.affine)
            A = np.asanyarray(pmap.dataobj)

            if sigma_mm>0:

                # multiply with a weight kernel representing the uncertain region
                # of interest around the coordinate
                kernel = create_gaussian_kernel(pmap,sigma_mm,sigma_truncation)
                r = int(kernel.shape[0]/2) # effective radius

                for i,xyzh in enumerate(XYZH):
                    xyz_vox = (np.dot(phys2vox,xyzh)+.5).astype('int')
                    x0,y0,z0 = [v-r for v in xyz_vox[:3]]
                    xs,ys,zs = [max(-v,0) for v in (x0,y0,z0)] # possible offsets
                    x1,y1,z1 = [min(xyz_vox[i]+r+1,A.shape[i]) for i in range(3)]
                    xd = x1-x0-xs
                    yd = y1-y0-ys
                    zd = z1-z0-zs
                    mapdata = A[x0+xs:x1,y0+ys:y1,z0+zs:z1] 
                    weights = kernel[xs:xs+xd,ys:ys+yd,zs:zs+zd]
                    assert(np.all(weights.shape==mapdata.shape))
                    prob = np.sum(np.multiply(weights,mapdata))
                    values[i].append(prob)

            else:
                # just read out the coordinate
                for i,xyzh in enumerate(XYZH):
                    xyz_vox = (np.dot(phys2vox,xyzh)+.5).astype('int')
                    x,y,z,_ = xyz_vox
                    values[i].append(A[x,y,z])


        matches = [{index:value for index,value in enumerate(P) if value>0} 
                    for P in values.values() ]

        assignments = [
            [(index,self.decode_label(index),value) for index,value
            in sorted(M.items(),key=lambda item:item[1],reverse=True)]
            for M in matches ]

        return assignments
