import logging
import nibabel as nib
from nibabel.affines import apply_affine
from numpy import linalg as npl
import numpy as np
import json
from collections import defaultdict
from functools import lru_cache

from brainscapes.region import construct_tree, Region
from brainscapes.retrieval import download_file
from brainscapes.registry import Registry,create_key
from brainscapes import parcellations, spaces
from brainscapes.space import Space
from brainscapes import features 

class Atlas:

    def __init__(self,identifier,name):
        # Setup an empty. Use _add_space and _add_parcellation to complete
        # the setup.
        self.name = name
        self.id = identifier
        self.key = create_key(name)
        self.regiontree = None
        self.features = defaultdict(list)
        self.parcellations = [] # add with _add_parcellation
        self.spaces = [] # add with _add_space

        # nothing selected yet at construction time
        self.selected_region = None
        self.selected_parcellation = None 

    def _add_space(self, space):
        self.spaces.append(space)

    def _add_parcellation(self, parcellation, select=False):
        self.parcellations.append(parcellation)
        if self.selected_parcellation is None or select:
            self.select_parcellation(parcellation)

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
        """
        Provides an object hook for the json library to construct an Atlas
        object from a json stream.
        """
        if all([ '@id' in obj, 'spaces' in obj, 'parcellations' in obj,
            obj['@id'].startswith("juelich/iav/atlas/v1.0.0") ]):
            p = Atlas(obj['@id'], obj['name'])
            for space_id in obj['spaces']:
                assert(space_id in spaces)
                p._add_space( spaces[space_id] )
            for parcellation_id in obj['parcellations']:
                assert(parcellation_id in parcellations)
                p._add_parcellation( parcellations[parcellation_id] )
            return p
        return obj

    def select_parcellation(self, parcellation):
        """
        Select a different parcellation for the atlas.

        Parameters
        ----------

        parcellation : Parcellation
            The new parcellation to be selected
        """
        if parcellation not in self.parcellations:
            logging.error('The requested parcellation is not supported by the selected atlas.')
            logging.error('    Parcellation:  '+parcellation['name'])
            logging.error('    Atlas:         '+self.name)
            logging.error(parcellation.id,self.parcellations)
            raise Exception('Invalid Parcellation')
        self.selected_parcellation = parcellation
        self.regiontree = construct_tree(parcellation)

    def get_maps(self, space):
        """
        Get the volumetric maps for the selected parcellation in the requested
        template space. Note that this sometimes included multiple Nifti
        objects. For example, the Julich-Brain atlas provides two separate
        maps, one per hemisphere.

        Parameters
        ----------
        space : template space 

        Yields
        ------
        A dictionary of nibabel Nifti objects representing the volumetric map.
        The key of each object is a string that indicates which part of the
        brain each map describes, and may be used to identify the proper
        region name. In case of Julich-Brain, for example, it is "left
        hemisphere" and "right hemisphere".
        """
        if space.id not in self.selected_parcellation.maps.keys():
            logging.error('The selected atlas parcellation is not available in the requested space.')
            logging.error('    Selected parcellation: {}'.format(self.selected_parcellation.name))
            logging.error('    Requested space:       {}'.format(space))
            return None
        print('Loading 3D map for space ', space)
        mapurl = self.selected_parcellation.maps[space.id]

        maps = {}
        if type(mapurl) is dict:
            # Some maps are split across multiple files, e.g. separated per
            # hemisphere. They are then given as a dictionary, where the key
            # represents a string that allows to identify them with region name
            # labels.
            for label,url in mapurl.items():
                filename = download_file(url)
                if filename is not None:
                    maps[label] = nib.load(filename)
        else:
            filename = download_file(mapurl)
            maps[''] = nib.load(filename)
        
        return maps

    def get_mask(self, space : Space):
        """
        Returns a binary mask  in the given space, where nonzero values denote
        voxels corresponding to the current region selection of the atlas. 

        WARNING: Note that for selections of subtrees of the region hierarchy, this
        might include holes if the leaf regions are not completly covering
        their parent and the parent itself has no label index in the map.

        Parameters
        ----------
        space : Space
            Template space 
        """
        # remember that some parcellations are defined with multiple / split maps
        return self._get_regionmask(space,self.selected_region)

    @lru_cache(maxsize=5)
    def _get_regionmask(self,space : Space,regiontree : Region):
        """
        Returns a binary mask  in the given space, where nonzero values denote
        voxels corresponding to the union of regions in the given regiontree.

        WARNING: Note that this might include holes if the leaf regions are not
        completly covering their parent and the parent itself has no label
        index in the map.

        NOTE: Function uses lru_cache, so if called repeatedly on the same
        space it will not recompute the mask.

        Parameters
        ----------
        space : Space 
            Template space 
        regiontree : Region
            A region from the region hierarchy (could be any of the root, a
            subtree, or a leaf)
        """
        print("Computing the mask for {} in {}".format(
            regiontree.name, space))
        maps = self.get_maps(space)
        mask = affine = header = None 
        for m in maps.values():
            D = np.array(m.dataobj)
            if mask is None: 
                # copy metadata for output mask from the first map!
                mask = np.zeros_like(D)
                affine, header = m.affine, m.header
            for r in regiontree.iterate():
                if 'labelIndex' not in r.attrs.keys():
                    continue
                if r.attrs['labelIndex'] is None:
                    continue
                mask[D==int(r.attrs['labelIndex'])]=1
        return nib.Nifti1Image(dataobj=mask,affine=affine,header=header)

    def get_template(self, space, resolution_mu=0, roi=None):
        """
        Get the volumetric reference template image for the given space.

        Parameters
        ----------
        space : str
            Template space definition, given as a dictionary with an '@id' key
        resolution :  float
            Desired target pixel spacing in micrometer (default: native spacing)
        roi : n/a
            3D region of interest (not yet implemented)

        TODO model the MNI URLs in the space definition

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        if space not in self.spaces:
            logging.error('The selected atlas does not support the requested reference space.')
            logging.error('    Requested space:       {}'.format(space.name))
            return None

        print('Loading template image for space',space.name)
        filename = download_file( space.url, space.ziptarget )
        if filename is not None:
            return nib.load(filename)
        else:
            return None

    def select_region(self,region):
        """
        Selects a particular region. 

        TODO test carefully for selections of branching points in the region
        hierarchy, then managing all regions under the tree. This is nontrivial
        because for incomplete parcellations, the union of all child regions
        might not represent the complete parent node in the hierarchy.

        Parameters
        ----------
        region : Region
            Region to be selected. Both a region object, as well as a region
            key (uppercase string identifier) are accepted.

        Yields
        ------
        True, if selection was successful, otherwise False.
        """
        searchname = region.key if isinstance(region,Region) else region
        selected = self.regiontree.find(searchname,search_key=True)
        if selected is not None:
            self.selected_region = selected
            logging.info('Selected region {}'.format(self.selected_region.name))
            return True
        else:
            return False

    def region_selected(self,region):
        """
        Verifies wether a given region is part of the current selection.
        """
        return self.selected_region.includes(region)

    def coordinate_selected(self,space,coordinate):
        """
        Verifies wether a position in the given space is inside the current
        selection.

        Parameters
        ----------
        space : Space
            The template space in which the test shall be carried out
        coordinate : tuple x/y/z
            A coordinate position given in the physical space. It will be
            converted to the voxel space using the inverse affine matrix of the
            template space for the query.

        NOTE: since get_mask is lru-cached, this is not necessary slow
        """
        assert(space in self.spaces)
        # transform physical coordinates to voxel coordinates for the query
        mask = self.get_mask(space)
        voxel = apply_affine(npl.inv(mask.affine),coordinate).astype(int)
        if np.any(voxel>=mask.dataobj.shape):
            return False
        if mask.dataobj[voxel[0],voxel[1],voxel[2]]==0:
            return False
        return True

    def query_data(self,modality,**kwargs):
        """
        Query data features for the currently selected region(s) by modality. 
        See brainscapes.features.modalities for available modalities.
        """
        assert(modality in features.modalities)
        hits = []
        for Pool in features.pools[modality]:
            if modality=='GeneExpression':
                pool = Pool(kwargs['gene'])
            else:
                pool = Pool()
            hits.extend(pool.pick_selection(self))
        return hits

    def connectivity_sources(self):
        #TODO refactor, this is dirty
        return [f['name'] for f in self.features['Connectivity Profiles']]

    def connectivity_matrix(self, srcname):
        """
        Tries to find a connectivity feature source with the given name, and
        construct a connectivity matrix from it.

        Parameters
        ----------
        srcname : str
            Name of a connectivity source, as listed by connectivity_sources()
        
        Yields
        ------
        A numpy object representing a connectivity matrix for the given parcellation, or None 
        """
        # TODO refactor, this is dirty
        for f in self.features['Connectivity Profiles']:
            if f['name'] == srcname:
                dim = len(f['data']['field names'])
                result = np.zeros((dim,dim))
                for i,field in enumerate(f['data']['field names']):
                    result[i,:] = f['data']['profiles'][field]
                return result
        raise Exception('No connectivity feature source found with the given name "{}"'.format(
            srcname))

    def connectivity_filter(self, src=None):
        print('connectivity filter for src: ' + src)
        #TODO implement


REGISTRY = Registry(
        'brainscapes.definitions.atlases', Atlas.from_json )

if __name__ == '__main__':

    atlas = REGISTRY.MULTILEVEL_HUMAN_ATLAS

    # atlas.get_maps('mySpace')
    # atlas.get_template("template")
    print(atlas.regiontree)
    print('*******************************')
    print(atlas.regiontree.find('hOc1'))
    print('*******************************')
    print(atlas.regiontree.find('LB (Amygdala) - left hemisphere'))
    print('******************************')
    print(atlas.regiontree.find('Ch 123 (Basal Forebrain) - left hemisphere'))
