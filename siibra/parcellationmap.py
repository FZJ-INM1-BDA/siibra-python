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
from . import logger,arrays
from .space import Space,SpaceVOI
from .commons import MapType,ParcellationIndex
from .volumesrc import ImageProvider,VolumeSrc
from .arrays import create_homogeneous_array,create_gaussian_kernel,argmax_dim4
from .region import Region

import numpy as np
import nibabel as nib
from nilearn import image
from memoization import cached
from tqdm import tqdm
from abc import abstractmethod
from typing import Union

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
        obj = classes[maptype](parcellation,space)
    elif maptype is None:
        logger.warning('No maptype provided when requesting the parcellation map. Falling back to MapType.LABELLED')
        obj = classes[MapType.LABELLED](parcellation,space)
    else:
        raise ValueError(f"Invalid maptype: '{maptype}'")
    if len(obj)==0:
        raise ValueError(f"No data found to construct a {maptype} map for {parcellation.name} in {space.name}.")
    
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

    def fetch(self,resolution_mm:float=None,mapindex:int=0, voi:SpaceVOI=None):
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

    def decode_label(self,mapindex=None,labelindex=None):
        """
        Decode the region associated to a particular index.

        Parameters
        ----------
        mapindex : Sequential index of the 3D map used, if more than one are included
        labelindex : Label index of the region, if the map is a labelled volume 
        """
        pindex = ParcellationIndex(map=mapindex,label=labelindex)
        region = self.regions.get(pindex)
        if region is None:
            raise ValueError(f"Could not decode parcellation index {pindex}")
        else:
            return region

    
    def decode_region(self,regionspec:Union[str,Region]):
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
        region = self.parcellation.decode_region(regionspec) if isinstance(regionspec,str) else regionspec
        result = []
        for idx,r in self.regions.items():
            if r==region:
                return [idx] 
            elif r.has_parent(region):
                result.append(idx)
        if len(result)==0:
            raise IndexError(f"Could not decode region specified by {regionspec} in {self.parcellation.name}")
        return result

    def extract_regionmap(self,regionspec:Union[str,int,Region],resolution_mm=None,voi:SpaceVOI=None):
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
            mapimg = self.fetch(resolution_mm=resolution_mm, mapindex=index.map,voi=voi)
            if index.label is not None:
                mapimg = nib.Nifti1Image(
                    dataobj=(np.asarray(mapimg.dataobj)==index.label).astype(np.uint8),
                    affine=mapimg.affine)
            mapimgs.append(mapimg)
        if len(mapimgs)==1:
            return mapimgs[0]
        elif self.maptype==MapType.LABELLED:
            m = mapimgs[0]
            for m2 in mapimgs[1:]:
                m.dataobj[m2.dataobj>0]=1
            return m
        else:
            logger.info(f"4D volume with {len(mapimgs)} continuous region maps extracted from region specification '{regionspec}'")
            return image.concat_imgs(mapimgs)
    
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
            unmatched = []
            for labelindex in np.unique(np.asarray(m.dataobj)):
                if labelindex!=0:
                    pindex = ParcellationIndex(map=mapindex,label=labelindex)
                    try:
                        region = self.parcellation.decode_region(pindex)
                        if labelindex>0:
                            self._regions_cached[pindex] = region
                    except ValueError:
                        unmatched.append(pindex)

            if unmatched:
                logger.warning(f"{len(unmatched)} parcellation indices in labelled volume couldn't be matched to region definitions in {self.parcellation.name}")
            
    @cached
    def _load_map(self,volume_src:VolumeSrc,resolution_mm:float,voi:SpaceVOI):
        m = volume_src.fetch(resolution_mm=resolution_mm,voi=voi)
        if len(m.dataobj.shape)==4 and m.dataobj.shape[3]>1:
            logger.info(f"{m.dataobj.shape[3]} continuous maps given - using argmax to generate a labelled volume. ")
            m = argmax_dim4(m)
        if m.dataobj.dtype.kind=='f':
            logger.warning(f"Floating point image type encountered when building a labelled volume for {self.parcellation.name}, converting to integer.")
            m = nib.Nifti1Image(dataobj=np.asarray(m.dataobj,dtype=int),affine=m.affine)
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
            assert(region.index.label)
            # load region mask
            mask_ = self._load_regional_map(region,resolution_mm=resolution_mm,voi=voi)
            if not mask_:
                continue
            # build up the aggregated mask with labelled indices
            if mask_.shape!=m.shape:
                mask = image.resample_to_img(mask_,m,interpolation='nearest')
            else:
                mask = mask_
            m.dataobj[mask.dataobj>0] = region.index.label

        return m


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
                pindex = ParcellationIndex(map=i,label=None)
                self._regions_cached[pindex] = region

            # we are finished, no need to look for regional map.
            return
                
        # otherwise we look for continuous maps associated to individual regions
        regions = [r for r in self.parcellation.regiontree 
                if r.has_regional_map(self.space,MapType.CONTINUOUS)]
        logger.info(f'{len(regions)} regional continuous maps found for {self.parcellation}.')
        for i,region in enumerate(regions):
            self._maploaders_cached.append(
                lambda r=region,res=None,voi=None:self._load_regional_map(r,resolution_mm=res,voi=voi))
            if region in self.regions.values():
                logger.debug(f"Region already seen in tree: {region.key}")
            pindex = ParcellationIndex(map=i,label=None)
            self._regions_cached[pindex] = region

    @cached
    def assign_coordinates(self,xyz_phys,sigma_mm=1,sigma_truncation=3):
        """
        Assign regions to a physical coordinates with optional standard deviation.

        Parameters
        ----------
        xyz_phys : 3D coordinate tuple, list of 3D tuples, or Nx3 array of coordinate tuples
            3D point(s) in physical coordinates of the template space of the
            ParcellationMap
        sigma_mm : float (default: 1)
            standard deviation /expected localization accuracy of the point, in
            mm units. A 3D Gaussian distribution with that
            bandwidth will be used for representing the location.
        sigma_truncation : float (default: 3)
            If sigma_phys is nonzero, this factor is used to determine where to
            truncate the Gaussian kernel in standard error units.
        """
        assert(sigma_mm>=1)

        # Convert input to Nx4 list of homogenous coordinates
        assert(len(xyz_phys)>0)
        XYZH = create_homogeneous_array(xyz_phys)
        numpts = XYZH.shape[0]

        logger.info((f"Assigning {numpts} uncertain coordinates (stderr={sigma_mm}) to {len(self)} maps." ))
        tpl = self.space.get_template().fetch()
        phys2vox = np.linalg.inv(tpl.affine)
        scaling = np.array([np.linalg.norm(tpl.affine[:,i]) for i in range(3)]).mean()
        kernel = create_gaussian_kernel(sigma_mm/scaling,sigma_truncation)
        r = int(kernel.shape[0]/2) # effective radius

        assignments = []
        for i,xyzh in enumerate(XYZH):
            xyz_vox = (np.dot(phys2vox,xyzh)+.5).astype('int')
            shift = np.identity(4)
            shift[:3,-1] = xyz_vox[:3]-r
            W = nib.Nifti1Image(dataobj=kernel,affine=np.dot(tpl.affine,shift))
            assignments.append(self.assign(W,msg=", ".join([f"{v:.1f}" for v in xyzh[:3]])))

        return assignments

    
    def assign(self,other,msg=None):
        
        values = {}
        if msg is None:
            msg=f"Assigning structure to {len(self)} maps"

        # crop the input and define its voi in reference space
        bb_vox = arrays.bbox3d(other.dataobj)
        minpt,maxpt = [v.ravel()[:3] for v in np.hsplit(bb_vox,2)]
        (x0,y0,z0),(x1,y1,z1) =  minpt,maxpt
        shift = np.eye(4)
        shift[:3,-1] = minpt.ravel()[:3]
        bb_mm = np.dot(other.affine,bb_vox)
        other_cropped = nib.Nifti1Image(
            dataobj = other.dataobj[x0:x1,y0:y1,z0:z1],
            affine = np.dot(other.affine,shift) )

        minpt,maxpt = np.hsplit(bb_mm,2)
        voi = self.space.get_voi(minpt,maxpt)

        resampled = None
        for mapindex,loadfnc in tqdm(enumerate(self.maploaders),total=len(self),desc=msg,unit=" maps"):

            this_cropped = loadfnc(voi=voi)
            if not this_cropped:
                logger.warning(f"Could not load regional map for {self.regions[mapindex].name}")
                continue
            if resampled is None:
                resampled = image.resample_to_img(other_cropped,this_cropped)
            value =  np.sum(np.multiply(resampled.dataobj,this_cropped.dataobj))
            if value>0:
                values[mapindex] = value

        pindex = lambda i: ParcellationIndex(map=i,label=None)
        assignments = [(pindex(i),self.decode_label(mapindex=i),value) 
                    for i,value in sorted(values.items(),key=lambda item:item[1],reverse=True)]
        return assignments