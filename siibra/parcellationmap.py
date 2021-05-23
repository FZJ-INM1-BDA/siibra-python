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

from . import logger
from .space import Space,SpaceVOI
from .commons import MapType,ImageProvider
from .region import Region

import numpy as np
from scipy.ndimage.measurements import sum_labels

import nibabel as nib
from nilearn import image
from memoization import cached
from scipy.ndimage import gaussian_filter
import numbers
from tqdm import tqdm
from abc import ABC, abstractmethod

def _assert_homogeneous_3d(xyz):
    if len(xyz)==4:
        return xyz
    else:
        return np.r_[xyz,1]

def _nifti_argmax_dim4(img,dim=-1):
    """
    Given a nifti image object with four dimensions, returns a modified object
    with 3 dimensions that is obtained by taking the argmax along one of the
    four dimensions (default: the last one). To distinguish the pure background
    voxels from the foreground voxels of channel 0, the argmax indices are
    incremented by 1 and label index 0 is kept to represent the background.
    """
    assert(len(img.shape)==4)
    assert(dim>=-1 and dim<4)
    newarr = np.asarray(img.dataobj).argmax(dim)+1
    # reset the true background voxels to zero
    newarr[np.asarray(img.dataobj).max(dim)==0]=0
    return nib.Nifti1Image(
            dataobj = newarr,
            header = img.header,
            affine = img.affine )


def _roiimg(refimg,xyz_phys,sigma_phys=1,sigma_point=3,resample=True):
    """
    Compute a region of interest heatmap with a Gaussian kernel 
    at the given position in physical coordinates corresponding 
    to the given template image. The output is a 3D spatial image
    with the same dimensions and affine as the template, including
    the heatmap.
    """
    xyzh = _assert_homogeneous_3d(xyz_phys)

    # position in voxel coordinates
    phys2vox = np.linalg.inv(refimg.affine)
    xyz_vox = (np.dot(phys2vox,xyzh)+.5).astype('int')
    
    # to the parcellation map in voxel space, given the physical kernel width.
    scaling = np.array([np.linalg.norm(refimg.affine[:,i]) 
                        for i in range(3)]).mean()
    sigma_vox = sigma_phys / scaling
    r = int(sigma_point*sigma_vox)
    k_size = 2*r + 1
    impulse = np.zeros((k_size,k_size,k_size))
    impulse[r,r,r] = 1
    kernel = gaussian_filter(impulse, sigma_vox)
    kernel /= kernel.sum()
    
    # compute the affine matrix for the kernel
    r = int(kernel.shape[0]/2)
    xs,ys,zs,_ = [v-r for v in xyz_vox]
    shift = np.array([
        [1,0,0,xs],
        [0,1,0,ys],
        [0,0,1,zs],
        [0,0,0,1]
    ])
    affine = np.dot(refimg.affine,shift)
    
    # create the resampled output image
    roiimg = nib.Nifti1Image(kernel,affine=affine)
    return image.resample_to_img(roiimg,refimg) if resample else roiimg

def _kernelimg(refimg,sigma_phys=1,sigma_point=3):
    """
    Compute a 3D Gaussian kernel for the voxel space of the given reference
    image, matching its bandwidth provided in physical coordinates.
    """
    scaling = np.array([np.linalg.norm(refimg.affine[:,i]) 
                        for i in range(3)]).mean()
    sigma_vox = sigma_phys / scaling
    r = int(sigma_point*sigma_vox)
    k_size = 2*r + 1
    impulse = np.zeros((k_size,k_size,k_size))
    impulse[r,r,r] = 1
    kernel = gaussian_filter(impulse, sigma_vox)
    kernel /= kernel.sum()
    
    return kernel


# Some parcellation maps require special handling to be expressed as a static
# parcellation. This dictionary contains postprocessing functions for converting
# the image objects returned when loading the map of a specific parcellations,
# in order to convert them to a 3D statis map. The dictionary is indexed by the
# parcellation ids.
_STATIC_MAP_HOOKS = { 
        parcellation_id : lambda img : _nifti_argmax_dim4(img)
        for  parcellation_id in [
            "minds/core/parcellationatlas/v1.0.0/d80fbab2-ce7f-4901-a3a2-3c8ef8a3b721",
            "minds/core/parcellationatlas/v1.0.0/73f41e04-b7ee-4301-a828-4b298ad05ab8",
            "minds/core/parcellationatlas/v1.0.0/141d510f-0342-4f94-ace7-c97d5f160235",
            "minds/core/parcellationatlas/v1.0.0/63b5794f-79a4-4464-8dc1-b32e170f3d16",
            "minds/core/parcellationatlas/v1.0.0/12fca5c5-b02c-46ce-ab9f-f12babf4c7e1" ]
        } 

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
        self.maploaders = []
        self._define_maploaders()

    @property
    def regions(self):
        """
        Dictionary of regions associated to the parcellion map, indexed by (labelindex,mapindex).
        Lazy implementation - self._link_regions() will be called when the regions are accessed for the first time.
        """
        if self._regions_cached is None:
            self._link_regions()
        return self._regions_cached

    @abstractmethod
    def _link_regions():
        pass

    @abstractmethod
    def _define_maploaders(self):
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

    def _define_maploaders(self):

        self.maploaders=[]

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
                self.maploaders.append(lambda res=None,voi=None: self._collect_maps(resolution_mm=res,voi=voi))
            elif source.volume_type==self.space.type:
                self.maploaders.append(lambda res=None,s=source,voi=None: self._load_maps(s,resolution_mm=res,voi=voi))

    def _link_regions(self):

        self._regions_cached = {}
        for mapindex,maploader in enumerate(self.maploaders):

            # load map at lowest resolution
            m = maploader()
            assert(m is not None)
            if m.dataobj.dtype.kind!='u':
                logger.warning(f'Labelled volumes expect unsigned integer type, but fetched image data is "{m.dataobj.dtype}". Will convert to int.')
                m = nib.Nifti1Image(np.asanyarray(m.dataobj).astype('uint'),m.affine)
            
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

    def _ensure_integertype(self,labelmap):
        if labelmap is None:
            return None
        elif labelmap.dataobj.dtype.kind!='u':
            logger.warning(f'Parcellation maps expect unsigned integer type, but the fetched image data has type "{labelmap.dataobj.dtype}". Will convert to int.')
            return nib.Nifti1Image(np.asanyarray(labelmap.dataobj).astype('uint'),labelmap.affine)
        else:
            return labelmap
            
    def _load_maps(self,volume_src,resolution_mm,voi):
        m = self._ensure_integertype(volume_src.fetch(resolution_mm,voi=voi))
        # apply postprocessing hook, if applicable
        if self.parcellation.id in _STATIC_MAP_HOOKS.keys():
            m = _STATIC_MAP_HOOKS[self.parcellation.id](m)
        if m.dataobj.dtype.kind!='u':
            raise RuntimeError("When loading a labelled volume, unsigned integer types are expected. However, this image data has type '{}'".format(
                m.dataobj.dtype))
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
            mask_ = self._ensure_integertype(self._load_regional_map(region,resolution_mm=resolution_mm,voi=voi))
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

    def _link_regions(self):
        self._regions_cached={}
        regions = [r for r in self.parcellation.regiontree 
                if r.has_regional_map(self.space,MapType.CONTINUOUS)]
        labelindex = -1
        for mapindex,region in enumerate(regions):
            if region in self.regions.values():
                logger.debug(f"Region already seen in tree: {region.key}")
                continue
            self._regions_cached[labelindex,mapindex] = region
    
    def _define_maploaders(self):
        self.maploaders=[]
        regions = [r for r in self.parcellation.regiontree 
                if r.has_regional_map(self.space,MapType.CONTINUOUS)]
        for region in regions:
            self.maploaders.append(
                lambda r=region,res=None,voi=None:self._load_regional_map(r,resolution_mm=res,voi=voi))

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
    def assign_regions(self,xyz_phys,sigma_phys=0,sigma_point=3,thres_percent=1,print_report=True):
        """
        Assign regions to a physical coordinates with optional standard deviation.

        TODO  allow to process multiple xyz coordinates at once

        Parameters
        ----------
        xyz_phys : 3D coordinate tuple, list of 3D tuples, or Nx3 array of coordinate tuples
            3D point(s) in physical coordinates of the template space of the
            ParcellationMap
        sigma_phys : float (default: 0)
            standard deviation /expected localization accuracy of the point, in
            physical units. If nonzero, A 3D Gaussian distribution with that
            bandwidth will be used for representing the location instead of a
            deterministic coordinate.
        sigma_point : float (default: 3)
            If sigma_phys is nonzero, this factor is used to determine where to
            truncate the Gaussian kernel in standard error units.
        thres_percent : float (default: 1)
            Regions with a probability below this threshold will not be returned.
        print_report : Boolean (default: True)
            Wether to print a short report to stdout
        """

        # Convert input to Nx4 list of homogenous coordinates
        assert(len(xyz_phys)>0)
        if isinstance(xyz_phys[0],numbers.Number):
            # only a single point provided
            assert(len(xyz_phys) in [3,4])
            XYZH = np.ones((1,4))
            XYZH[0,:len(xyz_phys)] = xyz_phys
        else:
            XYZ = np.array(xyz_phys)
            assert(XYZ.shape[1]==3)
            XYZH = np.c_[XYZ,np.ones_like(XYZ[:,0])]
        numpts = XYZH.shape[0]

        if sigma_phys>0:
            logger.info((
                f"Performing assignment of {numpts} uncertain coordinates "
                f"(stderr={sigma_phys}) to {len(self)} maps." ))
        else:
            logger.info((
                f"Performing assignment of {numpts} deterministic coordinates "
                f"to {len(self)} maps."))

        probs = {i:[] for i in range(numpts)}
        for mapindex,loadfnc in tqdm(enumerate(self.maploaders),total=len(self)):

            pmap = loadfnc(res=-1)
            assert(pmap.dataobj.dtype.kind=='f')
            if not pmap:
                logger.warning(f"Could not load regional map for {self.regions[-1,mapindex].name}")
                for i in range(numpts):
                    probs[i].append(-1)
                continue
            phys2vox = np.linalg.inv(pmap.affine)
            A = np.asanyarray(pmap.dataobj)

            if sigma_phys>0:

                # multiply with a weight kernel representing the uncertain region
                # of interest around the coordinate
                kernel = _kernelimg(pmap,sigma_phys,sigma_point)
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
                    probs[i].append(prob)

            else:
                # just read out the coordinate
                for i,xyzh in enumerate(XYZH):
                    xyz_vox = (np.dot(phys2vox,xyzh)+.5).astype('int')
                    x,y,z,_ = xyz_vox
                    probs[i].append(A[x,y,z])


        matches = [
                {self.decode_label(index):round(prob*100,2)
                    for index,prob in enumerate(P) 
                    if prob>0 }
                for P in probs.values() ]

        assignments = [
                [(region,prob_percent) for region,prob_percent
                    in sorted(M.items(),key=lambda item:item[1],reverse=True)
                    if prob_percent>=thres_percent]
                for M in matches ]

        if print_report:
            layout = "{:50.50} {:>12.12}"
            for i,assignment in enumerate(assignments):
                print()
                print(f"Assignment of location {XYZH[i,:3]} in {self.space.name}")
                print(layout.format("Brain region name","map value"))
                print(layout.format("-----------------","-----------"))
                for region,prob in assignment:
                    print(layout.format(region.name,prob))

        return assignments


