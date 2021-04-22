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

from . import logger, spaces, retrieval
from .space import Space
from .region import Region
from .bigbrain import BigBrainVolume,is_ngprecomputed,load_ngprecomputed
from .config import ConfigurationRegistry
from .commons import create_key
import numpy as np
import nibabel as nib
from nilearn import image
from enum import Enum
from tqdm import tqdm
from memoization import cached
from scipy.ndimage import gaussian_filter

class Parcellation:

    def __init__(self, identifier : str, name : str, version=None):
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.version = version
        self.publications = []
        self.description = ""
        self.maps = {}
        self.regiontree = Region(self.name,self)

    @cached
    def get_map(self, space: Space, resolution=None, regional=False, squeeze=True ):
        """
        Get the volumetric maps for the parcellation in the requested
        template space. This might in general include multiple 
        3D volumes. For example, the Julich-Brain atlas provides two separate
        maps, one per hemisphere. Per default, multiple maps are concatenated into a 4D
        array, but you can choose to retrieve a dict of 3D volumes instead using `return_dict=True`.

        Parameters
        ----------
        space : Space
            template space 
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.
        regional : Boolean (default: False)
            If True, will build a 4D map where each slice along
            the fourth dimension corresponds to a specific map of an individual
            region. Use this to request probability maps.
        squeeze : Boolean (default: True)
            If True, and if the fourth dimension of the resulting parcellation
            map is only one, will return a 3D volume image.

        Yields
        ------
        A ParcellationMap representing the volumetric map.
        """
        if space not in self.maps:
            raise ValueError('Parcellation "{}" does not provide a map for space "{}"'.format(
                str(self), str(space) ))

        maptype = ParcellationMap.MapType.REGIONAL_MAPS if regional else ParcellationMap.MapType.LABELLED_VOLUME
        return ParcellationMap(self,space,resolution=resolution, maptype=maptype, squeeze=squeeze)

    @property
    def labels(self):
        return self.regiontree.labels

    @property
    def names(self):
        return self.regiontree.names

    def supports_space(self,space):
        """
        Return true if this parcellation supports the given space, else False.
        """
        return space in self.maps.keys()

    def decode_region(self,regionspec):
        """
        Given a unique specification, return the corresponding region.
        The spec could be a label index, a (possibly incomplete) name, or a
        region object.
        This method is meant to definitely determine a valid region. Therefore, 
        if no match is found, it raises a ValueError. If it finds multiple
        matches, it tries to return only the common parent node. If there are
        multiple remaining parent nodes, which is rare, a custom group region is constructed.

        Parameters
        ----------
        regionspec : any of 
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key, 
            - an integer, which is interpreted as a labelindex,
            - a region object

        Return
        ------
        Region object
        """
        candidates = self.regiontree.find(regionspec,select_uppermost=True)
        if not candidates:
            raise ValueError("Regionspec {} could not be decoded under '{}'".format(
                regionspec,self.name))
        elif len(candidates)==1:
            return candidates[0]
        else:
            return Region._build_grouptree(candidates,self)


    def find_regions(self,regionspec):
        """
        Find regions with the given specification in this parcellation.

        Parameters
        ----------
        regionspec : any of 
            - a string with a possibly inexact name, which is matched both
              against the name and the identifier key, 
            - an integer, which is interpreted as a labelindex
            - a region object

        Yield
        -----
        list of matching regions
        """
        return self.regiontree.find(regionspec)


    def __str__(self):
        return self.name

    def __repr__(self):
        return  self.name

    def __eq__(self,other):
        """
        Compare this parcellation with other objects. If other is a string,
        compare to key, name or id.
        """
        if isinstance(other,Parcellation):
            return self.id==other.id
        elif isinstance(other,str):
            return any([
                self.name==other, 
                self.key==other,
                self.id==other])
        else:
            raise ValueError("Cannot compare object of type {} to Parcellation".format(type(other)))

    def __iter__(self):
        """
        Returns an iterator that goes through all regions in this parcellation
        """
        return self.regiontree.__iter__()

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct a Parcellation
        object from a json stream.
        """
        required_keys = ['@id','name','shortName','maps','regions']
        if any([k not in obj for k in required_keys]):
            return obj

        # create the parcellation, it will create a parent region node for the regiontree.
        version = obj['version'] if 'version' in obj else None
        p = Parcellation(obj['@id'], obj['shortName'], version)

        # add any children to the parent regiontree
        p.regiontree.children = tuple( 
                Region.from_json(regiondef,p) 
                for regiondef in obj['regions'] )

        p.maps = { spaces[space_id] : urls 
                for space_id, urls in obj['maps'].items() }

        if 'description' in obj:
            p.description = obj['description']
        if 'publications' in obj:
            p.publications = obj['publications']
        return p


def _assert_homogeneous_3d(xyz):
    if len(xyz)==4:
        return xyz
    else:
        return np.r_[xyz,1]


class ParcellationMap:

    """
    Represents a brain map in a reference space, with
    specific knowledge about the region information per labelindex or channel.
    Contains a Nifti1Image object as the "image" member.

    There are two types:
        1) Parcellation maps / labelled volumes
            A 3D or 4D volume with integer labels separating different,
            non-overlapping regions. The number of regions corresponds to the
            number of nonzero image labels in the volume.
        2) 4D overlapping regional maps (often probability maps).
            a 4D volume where each "time"-slice is a 3D volume representing
            a map of a particular brain region. This format is used for
            probability maps and similar continuous forms. The number of
            regions correspond to the z dimension of the 4 object.

    ParcellationMaps can be also constructred from neuroglancer (BigBrain) volumes if
    a feasible downsampled resolution is provided.

    TODO: For DiFuMo, implement a shortcut for computing the overlapping maps
    """

    class MapType(Enum):
        LABELLED_VOLUME = 1
        REGIONAL_MAPS = 2

    # Some parcellation maps require special handling to be expressed as a static
    # parcellation. This dictionary contains postprocessing functions for converting
    # the image objects returned when loading the map of a specific parcellations,
    # in order to convert them to a 3D statis map. The dictionary is indexed by the
    # parcellation ids.
    _STATIC_MAP_HOOKS = { 
            parcellation_id : lambda img : ParcellationMap._nifti_argmax_dim4(img)
            for  parcellation_id in [
                "minds/core/parcellationatlas/v1.0.0/d80fbab2-ce7f-4901-a3a2-3c8ef8a3b721",
                "minds/core/parcellationatlas/v1.0.0/73f41e04-b7ee-4301-a828-4b298ad05ab8",
                "minds/core/parcellationatlas/v1.0.0/141d510f-0342-4f94-ace7-c97d5f160235",
                "minds/core/parcellationatlas/v1.0.0/63b5794f-79a4-4464-8dc1-b32e170f3d16",
                "minds/core/parcellationatlas/v1.0.0/12fca5c5-b02c-46ce-ab9f-f12babf4c7e1" ]
            } 

    @staticmethod
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


    def __init__(self, parcellation: Parcellation, space: Space, maptype=MapType.LABELLED_VOLUME, resolution=None, squeeze=True):
        """
        Construct a ParcellationMap for the given parcellation and space.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        maptype : ParcellationMap.MapType
            The desired type of the map
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution if it is a
            neuroglancer high-resolution volume. 
        squeeze : Boolean (default: True)
            If True, and if the fourth dimension of the resulting parcellation
            map is only one, will only create a 3D volume image.
        """

        if space not in parcellation.maps:
            raise ValueError( 'Parcellation "{}" does not provide a map for space "{}"'.format(
                parcellation.name, space.name ))

        # load the image data
        if maptype==ParcellationMap.MapType.REGIONAL_MAPS:
            maps = ParcellationMap._load_regional_maps(parcellation,space,resolution)
        elif maptype==ParcellationMap.MapType.LABELLED_VOLUME:
            maps = ParcellationMap._load_parcellation_maps(parcellation,space,resolution)
        else:
            raise ValueError("Invalid maptype requested.")

        # determine region assignments to indices
        if maptype==ParcellationMap.MapType.LABELLED_VOLUME:
            self.regions = {
                    (labelindex,mapindex) : parcellation.decode_region(int(labelindex))
                    for mapindex,img in enumerate(maps.values())
                    for labelindex in np.unique(np.asarray(img.dataobj))
                    if labelindex>0 }
        else:
            self.regions = { (i,0) : region
                    for i,region in enumerate(maps.keys()) }

        # initialize the Nifti1Image with the retrieved data
        dtypes = list({m.dataobj.dtype for m in maps.values()})
        if len(dtypes)!=1:
            raise ValueError("dtypes not consistent in the different maps")
        dtype = dtypes[0]

        if len(maps)>1:
            logger.info('Concatenating {} 3D volumes into the final parcellation map...'.format(len(maps)))
            mapimg = image.concat_imgs(maps.values(),dtype=dtype)
            self.image = nib.Nifti1Image(mapimg.dataobj,mapimg.affine)
        else:
            mapimg = next(iter(maps.values()))
            if squeeze:
                self.image = nib.Nifti1Image(np.asanyarray(mapimg.dataobj).squeeze(),mapimg.affine)
            else:
                self.image = mapimg
        self.maptype = maptype
        self.space = space
        self.squeeze = squeeze
        self.parcellation = parcellation

    @staticmethod
    def _load_parcellation_maps(parcellation,space,resolution):
        """
        Load the parcellation map in the given space.
        maps.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution if it is a
            neuroglancer high-resolution volume. 

        Return
        ------
        maps : dict of Nifti1Images
            The found maps indexed by their corresponding region
        """

        maps = {}
        for mapindex,url in enumerate(parcellation.maps[space]):
            m = None
            if url=="collect":

                # build a 3D volume from the list of all regional maps
                logger.debug("Collecting labelled volume maps")
                regionmaps = ParcellationMap._load_regional_maps(parcellation,space,resolution)
                logger.debug("{} labelled volume maps found".format(len(regionmaps)))
                maplist = list(regionmaps.items())
                if len(maplist)==0: 
                    continue
                _,m = maplist[0]
                for region,mask_ in maplist:
                    # we also process maplist[0] again, so the labelindex is correct
                    assert(region.labelindex)
                    if mask_.shape != m.shape:
                        mask = image.resample_to_img(mask_,m,interpolation='nearest')
                    else:
                        mask = mask_
                    m.dataobj[mask.dataobj>0] = region.labelindex

            elif is_ngprecomputed(url):
                m = load_ngprecomputed(url,resolution)

            else:
                filename = retrieval.download_file(url)
                if filename is None:
                    continue
                m = nib.load(filename)
                if m.dataobj.dtype.kind!='u':
                    logger.warning('Parcellation maps expect unsigned integer type, but the fetched image data has type "{}". Will convert to int explicitly.'.format(m.dataobj.dtype))
                    m = nib.Nifti1Image(np.asanyarray(m.dataobj).astype('uint'),m.affine)

            # apply postprocessing hook, if applicable
            if parcellation.id in ParcellationMap._STATIC_MAP_HOOKS.keys():
                hook = ParcellationMap._STATIC_MAP_HOOKS[parcellation.id]
                m = hook(m)

            if m.dataobj.dtype.kind!='u':
                raise RuntimeError("When loading a labelled volume, unsigned integer types are expected. However, this image data has type '{}'".format(
                    m.dataobj.dtype))

            maps[mapindex] = m

        return maps


    @staticmethod
    def _load_regional_maps(parcellation:Parcellation, space:Space, resolution=None):
        """
        Traverse the parcellation object's regiontree to find region-specific
        maps.

        Parameters
        ----------
        parcellation : Parcellation
            The parcellation object used to build the map
        space : Space
            The desired template space to build the map
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution if it is a
            neuroglancer high-resolution volume. 

        Return
        ------
        maps : dict of Nifti1Images
            The found maps indexed by their corresponding region
        """

        regions = [r for r in parcellation.regiontree if r.has_regional_map(space)]
        if len(regions)>10:
            logger.info('Loading regional maps for {} regions, this may take a bit.'.format(len(regions)))

        maps = {}
        num_redundant = 0
        for r in tqdm(regions,total=len(regions)):
            logger.debug("Loading regional map for {}".format(r.name))
            regionmap = r.get_regional_map(space,quiet=True,resolution=resolution)
            if regionmap is None:
                logger.debug("not found")
                continue
            logger.debug("Loaded regional map for {}".format(r.name))
            if r in maps.keys():
                logger.debug("Region already seen in tree: {}".format(r.key))
                num_redundant += 1
            maps[r] = regionmap

        if num_redundant>0:
            logger.info("{} regions were redundant in the tree (possibly same child region anchored to multiple parent regions).".format(num_redundant))
        return maps

    # Forward a few Nifti1Image methods to this class.
    # Unfortunately, directly deriving from Nifti1Image did not work well
    # since nilearn performs some explicit type() checks internally, so it
    # turned out that many functions could not be applied to the derived class.

    @property
    def shape(self):
        return self.image.shape

    @property
    def dataobj(self):
        return self.image.dataobj

    @property
    def affine(self):
        return self.image.affine

    @property
    def dtype(self):
        return self.image.dataobj.dtype

    def as_image(self):
        """
        Expose this ParcellationMap as a Nifti1Image object.
        """
        return self.image

    @property
    def slicer(self):
        """
        Get numpy-style index slicing access to the internal Nifti1Image.
        """
        return self.image.slicer

    def __iter__(self):
        """
        Get an iterator along the fourth dimension of the parcellation map (if
        any), returning the 3D maps in order.
        """
        if len(self.shape)==4:
            return image.iter_img(self.image)
        else:
            # not much to iterate, this is a single 3D volume
            return iter((self.image,))

    def __contains__(self,spec):
        """
        Test if a 3D map identified by the given specification is included in this parcellation map. 
        For integer values, it is checked wether a corresponding slice along the fourth dimension could be extracted.
        Alternatively, a region object can be provided, and it will be checked wether the region is mapped.
        You might find the decode_region() function of Parcellation and Region objects useful for the latter.
        """
        if isinstance(spec,int):
            if len(self.image.shape)==3 and spec!=0:
                return False
            if len(self.image.shape)==4 and spec>=self.shape[3]:
                return False
            return True
        elif isinstance(spec,Region):
            for _,region in self.regions.items():
                if region==spec:
                    return True
        return False

    def __len__(self):
        return 1 if len(self.image.shape)==3 else self.image.shape[3]

    def __getitem__(self,index):
        """
        Get access to the different 3D maps included in this parcellation map, if any.
        For integer values, the corresponding slice along the fourth dimension at the given index is returned.
        Alternatively, a region object can be provided, and an attempt will be
        made to recover the index for this region.
        You might find the decode_region() function of Parcellation and Region objects useful for the latter.
        """
        
        # Try to convert the given index into a valid slice index
        sliceindex = None
        if isinstance(index,int):
            if any([ 
                len(self.image.shape)==3 and index==0, 
                len(self.image.shape)==4 and index<self.shape[3] ]):
                sliceindex=index
        elif isinstance(index,Region):
            for (labelindex,mapindex),region in self.regions.items():
                if region==index:
                    if self.maptype==ParcellationMap.MapType.LABELLED_VOLUME:
                        sliceindex = mapindex
                    else:
                        sliceindex = labelindex
        else:
            raise ValueError("Index access to ParcellationMap expects an Integer or Region object as index, but type {} was provided.".format(type(index)))

        # Fail if that wasn't successful
        if sliceindex is None:
            raise ValueError("Parcellation map of {} in {} cannot be indexed by '{}'".format(
                    self.parcellation.name, self.space.name, index))

        # We have a valid slice index! Return the requested slice.
        if len(self.image.shape)==3:
            assert(sliceindex==0)
            return self.image
        else:
            return self.image.slicer[:,:,:,sliceindex]

    def decode_region(self,index:int,map_index=0):
        """
        Decode the region associated to a particular index.
        for REGIONAL_MAPS, this is the index of the slice along the fourth dimension.
        For LABELLED_VOLUME types, this is the labelindex, ie. the color at a given voxel.
        For LABELLED_VOLUME types with multiple maps, the map index can be provided in addition.

        Parameters
        ----------
        index : int
            The index
        map_index : int (default=0)
            Index of the fourth dimension of a labelled volume with more than
            a single parcellation map.
        """
        return self.regions[index,map_index]

    def get_mask(self,index:int, map_index=0):
        """
        Extract the mask for one particular region. For parcellation maps, this
        is a binary mask volume. For overlapping maps, this is the
        corresponding slice, which typically is a volume of float type.

        Parameters
        ----------
        index : int
            For 4D (overlapping) maps, the index into the 4th dimension.
            For 3D maps, the label index of a region.
        map_index : int (default=0)
            Index of the fourth dimension of a labelled volume with more than
            a single parcellation map.
        """
        region = self.decode_region(index,map_index)
        assert(region)
        if self.maptype == ParcellationMap.MapType.LABELLED_VOLUME:
            map3d = self[map_index] 
            return nib.Nifti1Image(
                    dataobj=(np.asarray(map3d.dataobj)==index).astype(int),
                    affine=map3d.affine)
        else:
            return self[index]

    def _roiimg(self,xyz_phys,sigma_phys=1,sigma_point=3):
        """
        Compute a region of interest heatmap with a Gaussian kernel 
        at the given position in physical coordinates corresponding 
        to the given template image. The output is a 3D spatial image
        with the same dimensions and affine as the template, including
        the heatmap.
        """
        xyzh = _assert_homogeneous_3d(xyz_phys)
        tpl = self[0] 

        # position in voxel coordinates
        phys2vox = np.linalg.inv(tpl.affine)
        xyz_vox = (np.dot(phys2vox,xyzh)+.5).astype('int')
        
        # Compute a rasterized 3D Gaussian kernel to be applied 
        # to the parcellation map in voxel space, given the physical kernel width.
        scaling = np.array([np.linalg.norm(tpl.affine[:,i]) 
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
        affine = np.dot(tpl.affine,shift)
        
        # create the resampled output image
        roiimg = nib.Nifti1Image(kernel,affine=affine)
        return image.resample_to_img(roiimg,tpl)


    def assign_regions(self,xyz_phys,sigma_phys=0,sigma_point=3,thres_percent=1,print_report=True):
        """
        Assign regions to a physical coordinates with optional standard deviation.

        TODO  allow to process multiple xyz coordinates at once

        Parameters
        ----------
        xyz_phys : coordinate tuple 
            3D point in physical coordinates of the template space of the
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
        if any([
            self.maptype!=ParcellationMap.MapType.REGIONAL_MAPS,
            self.dtype.kind!='f' ]):
            raise NotImplementedError("Region assignment is only implemented for floating type regional maps for now.")

        logger.info("Performing assignment of 1 coordinate to {} maps.".format(len(self)))
        xyzh = _assert_homogeneous_3d(xyz_phys)

        if sigma_phys==0:
            
            phys2vox = np.linalg.inv(self.affine)
            xyz_vox = (np.dot(phys2vox,xyzh)+.5).astype('int')
            x,y,z,_ = xyz_vox
            probs = self.image.dataobj[x,y,z,:]

        else:

            # TODO it should be more efficient to no resample the roi above, 
            # but instead resample the pmap to the roi here. However, this didn't 
            # work when I tried it, maybe resample is instable if the target is much smaller?
            roi = self._roiimg(xyzh,sigma_phys,sigma_point=sigma_point)
            probs = []
            for pmap in tqdm(self,total=self.shape[3]):
                probs.append(np.sum(np.multiply(roi.dataobj,pmap.dataobj)))

        matches = {self.decode_region(index):round(prob*100,2)
                for index,prob in enumerate(probs) 
                if prob>0 }

        assignments = [(region,prob_percent) for region,prob_percent
                in sorted(matches.items(),key=lambda item:item[1],reverse=True)
                if prob_percent>=thres_percent]

        layout = "{:50.50} {:>12.12}"
        print()
        print(layout.format("Brain region name","Probability"))
        print(layout.format("-----------------","-----------"))
        for region,prob in assignments:
            print(layout.format(region.name,prob))

        return assignments

REGISTRY = ConfigurationRegistry('parcellations', Parcellation)
