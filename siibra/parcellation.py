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
import numbers
import numpy as np
import nibabel as nib
from nilearn import image
from enum import Enum
from tqdm import tqdm
from memoization import cached
from scipy.ndimage import gaussian_filter
from .volume_src import VolumeSrc

class Parcellation:

    def __init__(self, identifier : str, name : str, version=None):
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.version = version
        self.publications = []
        self.description = ""
        self.maps = {}
        self.volume_src = {}
        self.regiontree = Region(self.name,self)

    def get_volume_src(self, space: Space):
        """
        Get volumes sources for the parcellation in the requested template space.

        Parameters
        ----------
        space : Space
            template space

        Yields
        ------
        A list of volume sources
        """
        if space not in self.volume_src:
            raise ValueError('Parcellation "{}" does not provide volume sources for space "{}"'.format(
                str(self), str(space) ))
        return self.volume_src[space]

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

    def decode_region(self,regionspec,mapindex=None):
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
        mapindex : integer, or None (optional)
            Some parcellation maps are defined over multiple 3D parcellation
            volumes with overlapping labelindices (e.g. splitting the
            hemispheres). For those, the optional mapindex can be used to 
            further restrict the matching regions.

        Return
        ------
        Region object
        """
        candidates = self.regiontree.find(regionspec,select_uppermost=True,mapindex=mapindex)
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
        
        if 'volumeSrc' in obj:
            p.volume_src = { spaces[space_id] : {
                key : [
                    VolumeSrc.from_json(v_src) for v_src in v_srcs
                ] for key, v_srcs in key_vsrcs.items()
            } for space_id, key_vsrcs in obj['volumeSrc'].items() }

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

        self.maptype = maptype
        self.parcellation = parcellation
        self.space = space
        self.resolution = resolution

        # check for available maps per region
        self.maploaders = []
        self.regions = {} # indexed by (labelindex,mapindex)

        if maptype==ParcellationMap.MapType.LABELLED_VOLUME:
            for mapindex,url in enumerate(self.parcellation.maps[self.space]):
                regionmap = self._load_parcellation_map(url)
                if regionmap:
                    self.maploaders.append(lambda quiet=False,url=url:self._load_parcellation_map(url,quiet=quiet))
                    for labelindex in np.unique(np.asarray(regionmap.dataobj)):
                        if labelindex==0:
                            continue # this is the background only
                        region = self.parcellation.decode_region(int(labelindex),mapindex)
                        if labelindex>0:
                            self.regions[labelindex,mapindex] = region

        elif maptype==ParcellationMap.MapType.REGIONAL_MAPS:
            regions = [r for r in parcellation.regiontree if r.has_regional_map(space)]
            labelindex = -1
            for region in regions:
                if region in self.regions.values():
                    logger.debug(f"Region already seen in tree: {region.key}")
                    continue
                #regionmap = self._load_regional_map(region)
                self.maploaders.append(lambda quiet=False,region=region:self._load_regional_map(region,quiet=quiet))
                mapindex = len(self.maploaders)-1
                self.regions[labelindex,mapindex] = region

        else:
            raise ValueError("Invalid maptype requested.")

    def build_image(self):
        """
        Builds a full 3D or 4D Nifti1Image object from this parcellation map.
        Use with caution, this might get large!
        """
        if len(self)>1:
            logger.info(f'Concatenating {len(self)} 3D volumes into the final parcellation map...')
            mapimg = image.concat_imgs((fnc() for fnc in self.maploaders))
            return nib.Nifti1Image(mapimg.dataobj,mapimg.affine)
        else:
            return self.maploaders[0]()

    @cached
    def _load_parcellation_map(self,url,quiet=False):
        """
        Try to generate a 3D parcellation map from given url.

        Parameters
        ----------
        url : str
            map url as provided by a siibra parcellation configuration file
        quiet : Boolean (default: False)
            suppress output messages

        Return
        ------
        map : Nifti1Image, or None
            The found map, if any
        """
        m = None
        if url=="collect":

            # build a 3D volume from the list of all regional maps
            if not quiet:
                logger.debug("Collecting labelled volume maps")

            # collect all available region maps
            regions = [r for r in self.parcellation.regiontree 
                    if r.has_regional_map(self.space)]
            m = None
            for region in regions:
                assert(region.labelindex)

                # load region mask
                mask_ = self._load_regional_map(region)
                if not mask_:
                    continue
                if mask_.dataobj.dtype.kind!='u':
                    if not quiet:
                        logger.warning('Parcellation maps expect unsigned integer type, but the fetched image data has type "{}". Will convert to int explicitly.'.format(mask_.dataobj.dtype))
                    mask_ = nib.Nifti1Image(np.asanyarray(mask_.dataobj).astype('uint'),m.affine)

                # build up the aggregated mask with labelled indices
                if m is None:
                    m = mask_
                if mask_.shape!=m.shape:
                    mask = image.resample_to_img(mask_,m,interpolation='nearest')
                else:
                    mask = mask_
                m.dataobj[mask.dataobj>0] = region.labelindex

        elif is_ngprecomputed(url):
            m = load_ngprecomputed(url,self.resolution)

        else:
            filename = retrieval.download_file(url)
            if filename is not None:
                m = nib.load(filename)
                if m.dataobj.dtype.kind!='u':
                    if not quiet:
                        logger.warning('Parcellation maps expect unsigned integer type, but the fetched image data has type "{}". Will convert to int explicitly.'.format(m.dataobj.dtype))
                    m = nib.Nifti1Image(np.asanyarray(m.dataobj).astype('uint'),m.affine)

        if not m:
            return None

        # apply postprocessing hook, if applicable
        if self.parcellation.id in ParcellationMap._STATIC_MAP_HOOKS.keys():
            hook = ParcellationMap._STATIC_MAP_HOOKS[self.parcellation.id]
            m = hook(m)

        if m.dataobj.dtype.kind!='u':
            raise RuntimeError("When loading a labelled volume, unsigned integer types are expected. However, this image data has type '{}'".format(
                m.dataobj.dtype))

        return m

    @cached
    def _load_regional_map(self,region : Region, quiet=False):
        """
        Load a region-specific map

        Parameters
        ----------
        region : Region
            the requested region
        quiet : Boolean (default: False)
            suppress output messages

        Return
        ------
        maps : Nifti1Image, or None
            The found map, if any
        """
        if not quiet:
            logger.info(f"Loading regional map for {region.name} in {self.space.name}")
        regionmap = region.get_regional_map(self.space,quiet=quiet, resolution=self.resolution)
        if regionmap is None:
            return  None
        return regionmap

    def __iter__(self):
        """
        Get an iterator along the parcellation maps, returning 3D maps in
        order.
        """
        return (loadfunc() for loadfunc in self.maploaders)

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

    def __getitem__(self,spec):
        """
        Get access to the different 3D maps included in this parcellation map, if any.
        For integer values, the corresponding slice along the fourth dimension
        at the given index is returned.
        Alternatively, a region object can be provided, and an attempt will be
        made to recover the index for this region.
        You might find the decode_region() function of Parcellation and Region
        objects useful for the latter.
        """
        if not spec in self:
            raise ValueError(f"Index '{spec}' is not valid for this ParcellationMap.")
        
        # Try to convert the given index into a valid slice index
        # this should always be successful since we checked validity of the index above
        sliceindex = None
        if isinstance(spec,int):
            sliceindex=spec
        else:
            for (_,mapindex),region in self.regions.items():
                if region==spec:
                    sliceindex = mapindex
        if sliceindex is None:
            raise RuntimeError(f"Invalid index '{spec}' for accessing this ParcellationMap.")

        return self.maploaders[sliceindex]()

    def decode_region(self,index:int,mapindex=None):
        """
        Decode the region associated to a particular index.
        for REGIONAL_MAPS, this is the index of the slice along the fourth dimension.
        For LABELLED_VOLUME types, this is the labelindex, ie. the color at a given voxel.
        For LABELLED_VOLUME types with multiple maps, the map index can be provided in addition.

        Parameters
        ----------
        index : int
            The index
        mapindex : int, or None (default=None)
            Index of the fourth dimension of a labelled volume with more than
            a single parcellation map.
        """
        if self.MapType==ParcellationMap.MapType.LABELLED_VOLUME:
            return self.regions[index,mapindex]
        else:
            return self.regions[-1,index]

    def get_mask(self,region:Region):
        """
        Extract the mask for one particular region. For parcellation maps, this
        is a binary mask volume. For overlapping maps, this is the
        corresponding slice, which typically is a volume of float type.

        Parameters
        ----------
        region : Region
            The desired region.

        Return
        ------
        Nifti1Image, if found, otherwise None
        """
        if not region in self:
            return None
        if self.maptype == ParcellationMap.MapType.LABELLED_VOLUME:
            mapimg = self[region] 
            index = region.labelindex
            return nib.Nifti1Image(
                    dataobj=(np.asarray(mapimg.dataobj)==index).astype(int),
                    affine=mapimg.affine)
        else:
            return self[region]

    @staticmethod
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

    @staticmethod
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
        if self.maptype!=ParcellationMap.MapType.REGIONAL_MAPS:
            raise NotImplementedError("Region assignment is only implemented for floating type regional maps for now.")

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

            pmap = loadfnc(quiet=True)
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
                kernel = ParcellationMap._kernelimg(pmap,sigma_phys,sigma_point)
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
                {self.decode_region(index):round(prob*100,2)
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

REGISTRY = ConfigurationRegistry('parcellations', Parcellation)
