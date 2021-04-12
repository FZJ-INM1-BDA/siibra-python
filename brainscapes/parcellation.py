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
from .commons import create_key, Glossary
from .bigbrain import BigBrainVolume,is_ngprecomputed
from .config import ConfigurationRegistry
from collections import defaultdict
import numpy as np
import nibabel as nib
from nibabel.spatialimages import SpatialImage
from nilearn import image
from anytree import PreOrderIter
# TODO consider a custom cache implementation
from memoization import cached

def nifti_argmax_dim4(img,dim=-1):
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


# Some parcellation maps require special handling to be expressed as a static
# parcellation. This dictionary contains postprocessing functions for converting
# the image objects returned when loading the map of a specific parcellations,
# in order to convert them to a 3D statis map. The dictionary is indexed by the
# parcellation ids.
STATIC_MAP_HOOKS = {
        "minds/core/parcellationatlas/v1.0.0/d80fbab2-ce7f-4901-a3a2-3c8ef8a3b721": lambda img : nifti_argmax_dim4(img),
        "minds/core/parcellationatlas/v1.0.0/73f41e04-b7ee-4301-a828-4b298ad05ab8": lambda img : nifti_argmax_dim4(img),
        "minds/core/parcellationatlas/v1.0.0/141d510f-0342-4f94-ace7-c97d5f160235": lambda img : nifti_argmax_dim4(img),
        "minds/core/parcellationatlas/v1.0.0/63b5794f-79a4-4464-8dc1-b32e170f3d16": lambda img : nifti_argmax_dim4(img),
        "minds/core/parcellationatlas/v1.0.0/12fca5c5-b02c-46ce-ab9f-f12babf4c7e1": lambda img : nifti_argmax_dim4(img)
        }

def load_ngprecomputed(url,mip,force):
    """
    Creates a map by loading from a neuroglancer precomputed volume.
    """
    assert(type(mip)==int)
    V = BigBrainVolume(url)
    return V.Image(mip,force=force,clip=True)

def load_collect(space,regiontree,resolution,force):
    """
    Creates a map by collecting all available individual region maps in the
    given tree into a blank volume that fills the given space.

    Parameters
    ----------
    resolution : float or None (Default: None)
        Request the template at a particular physical resolution. If None,
        the native resolution is used.
        Currently, this only works for the BigBrain volume.
    """

    # initialize the empty map volume
    V = BigBrainVolume(space.url)
    mip = V.determine_mip(resolution)
    if mip is None:
        return None
    M = SpatialImage( 
            np.zeros(V.volume.mip_shape(mip),dtype=int), 
            affine=V.affine(mip) )

    # collect and add regional maps
    for region in PreOrderIter(regiontree):
        if "maps" not in region.attrs.keys():
            continue
        if space.id in region.attrs["maps"].keys():
            logger.info("Found a map for region '{}'".format(str(region)))
            assert("labelIndex" in region.attrs.keys())
            Mr0 = load_ngprecomputed(
                    region.attrs["maps"][space.id],mip,force)
            Mr = image.resample_to_img(Mr0,M)
            M.dataobj[Mr.dataobj>0] = int(region.attrs['labelIndex'])

    return M

def load_nifti(url):
    """
    Try to generate a map by loading a nifti from the given url.
    """
    filename = retrieval.download_file(url)
    if filename is None:
        return None
    return nib.load(filename)


class Parcellation:

    def __init__(self, identifier, name, version=None):
        self.id = identifier
        self.name = name
        self.key = create_key(name)
        self.version = version
        self.publications = []
        self.description = ""
        self.maps = defaultdict(dict)

        # rely on a dedicated call to construct_tree():
        self.regions = None
        self.regionnames = None

    def register_map(self, space_id, name, url):
        assert(space_id in spaces)
        self.maps[spaces[space_id]][name] = url

    def construct_tree(self,regions,parent=None):
        """ 
        Builds a complete tree recursively from a regions data structure.
        """
        if parent is None:
            self.regions = Region({'name':self.name},self)
            self.construct_tree(regions,parent=self.regions)
            self.regionnames = Glossary([c.key 
                for r in self.regions.iterate()
                for c in r.children ])
            self.labelindices = { c.labelIndex 
                    for r in self.regions.iterate()
                    for c in r.children 
                    if 'labelIndex' in c.attrs.keys() }
        else:
            for regiondef in regions:
                node = Region(regiondef,self,parent)
                if "children" in regiondef.keys():
                    #_ = self.construct_tree(
                    self.construct_tree(
                            regiondef['children'],parent=node)
            # inherit labelindex from children, if they agree
            if (parent.labelindex is None) and (parent.children is not None):
                L = [c.labelindex for c in parent.children]
                if (len(L)>0) and (L.count(L[0])==len(L)):
                    parent.labelindex = L[0]


    def get_maps(self, space: Space, tree: Region=None, resolution=None, force=False, return_dict=False):
        """
        Get the volumetric maps for the parcellation in the requested
        template space. Note that this might in general include multiple 
        3D volumes. For example, the Julich-Brain atlas provides two separate
        maps, one per hemisphere. Per default, these are concatenated into a 4D
        array in this case, but you can choose to retrieve a dict of 3D
        volumes.

        Parameters
        ----------
        space : Space
            template space 
        tree : Region (optional, default: None)
            if provided, only regions inside the region tree are included in the map.
            TODO this applies only to BigBrain for now
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.
        force : Boolean (default: False)
            if true, will start large downloads even if they exceed the download
            threshold set in the gbytes_feasible member variable (applies only
            to BigBrain space currently).
        return_dict : Boolean (default: False)
            If true, mulitple available maps will not be concatenated into a 4D
            array, but put returned as a dictionary of separate 3D objects.

        Yields
        ------
        A single SpatialImage, or dictionary of multiple SpatialImage objects
        representing the volumetric map.
        The key in the dictionary is a string that indicates which part of the
        brain each map describes, and may be used to identify the proper
        region name. In case of Julich-Brain, for example, it is "left
        hemisphere" and "right hemisphere".
        """
        not_avail_msg = 'Parcellation "{}" does not provide a map for space "{}"'.format(
                str(self), str(space) )
        if space not in self.maps:
            logger.error(not_avail_msg)
        if len(self.maps[space])==0:
            logger.error(not_avail_msg)


        maps = {}
        for label,url in self.maps[space].items():
            m = None
            if url=="collect":
                logger.info("This space has no complete map available. Will try to find individual area maps and aggregate them into one instead.")
                regiontree = self.regions if tree is None else tree
                m = load_collect(space,regiontree,resolution,force)
            elif is_ngprecomputed(url):
                vol = BigBrainVolume(space.url)
                mip = vol.determine_mip(resolution)
                if mip is not None:
                    m = load_ngprecomputed(url,mip,force)
                else:
                    logger.warn('Requested resolution invalid.')
            else:
                m = load_nifti(url)
            if m is not None:
                    maps[label] = m

        # apply postprocessing hook, if applicable
        if self.id in STATIC_MAP_HOOKS.keys():
            hook = STATIC_MAP_HOOKS[self.id]
            for k,v in maps.items():
                maps[k] = hook(v)

        if return_dict:
            return maps
        elif len(maps)==1:
            return next(iter(maps.values()))
        else:
            # concatenate multiple maps into 4D image
            dtypes = [c.dataobj.dtype for c in maps.values()]
            assert(all([d==dtypes[0] for d in dtypes]))
            return image.concat_imgs(maps.values(),dtype=dtypes[0])

    def find(self,region):
        """
        Search the regiontree.
        """
        return self.regions.find(region)

    @cached(max_size=10)
    def get_regionmask(self,space : Space, regiontree : Region, try_thres=None,force=False, resolution=None ):
        """
        Returns a binary mask where nonzero values denote
        voxels corresponding to the union of regions in the given regiontree.

        WARNING: Note that this might include holes if the leaf regions are not
        completly covering their parent and the parent itself has no label
        index in the map.

        NOTE: Function uses lru_cache, so if called repeatedly on the same
        space it will not recompute the mask.

        TODO passing a regiontree is a big akward since this cannot work with arbitrary regiontrees - we need to make sure it is a subtree of the parcellation that owns this map.

        TODO this has redundancies with Region.get_mask(), consider streamlining the two

        Parameters
        ----------
        regiontree : Region
            A region from the region hierarchy (could be any of the root, a
            subtree, or a leaf)
        try_thres : float or None,
            If defined, will prefere threshold continous maps for mask building, see self._threshold_continuous_map.
            We make this an explicit parameter to make lru_cache aware of using it.
        force : Boolean (default: False)
            if true, will start large downloads even if they exceed the download
            threshold set in the gbytes_feasible member variable (applies only
            to BigBrain space currently).
        resolution : float or None (Default: None)
            Request the template at a particular physical resolution. If None,
            the native resolution is used.
            Currently, this only works for the BigBrain volume.
        """

        not_avail_msg = 'Parcellation "{}" does not provide a map for space "{}"'.format(
                str(self), str(space) )
        if space not in self.maps:
            logger.error(not_avail_msg)
        if len(self.maps[space])==0:
            logger.error(not_avail_msg)

        logger.debug("Computing the mask for {} in {}".format(
            regiontree.name, space))
        maps = self.get_maps(space, tree=regiontree, force=force,
                resolution=resolution,return_dict=True)
        mask = affine = None 
        for description,m in maps.items():
            D = np.array(m.dataobj)
            if mask is None: 
                # copy metadata for output mask from the first map!
                mask = np.zeros_like(D)
                affine = m.affine
            for r in regiontree.iterate():
                if len(maps)>1 and (description not in r.name):
                    continue
                if 'labelIndex' not in r.attrs.keys():
                    continue
                if r.attrs['labelIndex'] is None:
                    continue

                # if enabled, check for available continuous maps that could be
                # thresholded instead of using the mask from the static
                # parcellation
                if try_thres is not None:
                    continuous_map = r.get_specific_map(space)
                    if continuous_map is not None:
                        logger.info('Using continuous map thresholded by {} for masking region {}.'.format(
                            try_thres, r))
                        mask[np.asarray(continuous_map.dataobj)>try_thres]=1
                        continue

                # in the default case, use the labelled area from the parcellation map
                mask[D==int(r.attrs['labelIndex'])]=1

        return SpatialImage(dataobj=mask,affine=affine)

    def __str__(self):
        return self.name

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

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct a Parcellation
        object from a json stream.
        """
        if '@id' in obj and 'maps' in obj:
            if 'version' in obj:
                p = Parcellation(obj['@id'], obj['name'], obj['version'])
            else:
                p = Parcellation(obj['@id'], obj['name'])
            for space_id,maps in obj['maps'].items():
                for name, url in maps.items():
                    p.register_map( space_id, name, url) 
            # TODO model the regions already here as a hierarchy tree
            if 'regions' in obj:
                p.construct_tree(obj['regions'])
            if 'description' in obj:
                p.description = obj['description']
            if 'publications' in obj:
                p.publications = obj['publications']
            return p
        return obj

REGISTRY = ConfigurationRegistry('parcellations', Parcellation)
