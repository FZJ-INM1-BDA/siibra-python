import logging
import nibabel as nib
import numpy as np
import json
from collections import defaultdict
from functools import lru_cache

from brainscapes import NAME2IDENTIFIER
from brainscapes.region import construct_tree
from brainscapes.definitions import atlases, parcellations, spaces, id2key, key2id
from brainscapes.retrieval import download_file

class Atlas:

    def __init__(self,identifier,name):
        # Setup an empty. Use _add_space and _add_parcellation to complete
        # the setup.
        self.name = name
        self.identifier = identifier
        self.regiontree = None
        self.features = defaultdict(list)
        self.parcellations = [] # add with _add_parcellation
        self.spaces = [] # add with _add_space

        # nothing selected yet at construction time
        self.selection = self.regiontree
        self.__parcellation__ = None 

    def _add_space(self, space_id):
        # TODO check that space_id has a valid object
        self.spaces.append(space_id)

    def _add_parcellation(self, parcellation_id, select=False):
        # TODO check that space_id has a valid object
        self.parcellations.append(parcellation_id)
        if self.__parcellation__ is None or select:
            self.select_parcellation(id2key(parcellation_id))

    def __str__(self):
        return self.name

    @staticmethod
    def from_json(obj):
        if all([ '@id' in obj, 'spaces' in obj, 'parcellations' in obj,
            obj['@id'].startswith("juelich/iav/atlas/v1.0.0") ]):
            p = Atlas(obj['@id'], obj['name'])
            for space_id in obj['spaces']:
                p._add_space( space_id )
            for parcellation_id in obj['parcellations']:
                p._add_parcellation( parcellation_id )
            return p
        return obj

    def select_parcellation(self, parcellationkey):
        """
        Select a different parcellation for the atlas.

        :param schema:
        """
        # TODO need more explicit formalization and testing of ontology
        # definition schemes
        assert(parcellationkey in parcellations)
        parcellation = parcellations[parcellationkey]
        if parcellation.id not in self.parcellations:
            logging.error('The requested parcellation is not supported by the selected atlas.')
            logging.error('    Parcellation:  '+parcellation['name'])
            logging.error('    Atlas:         '+self.name)
            logging.error(parcellationobj['@id'],self._parcellations)
            raise Exception('Invalid Parcellation')
        self.__parcellation__ = parcellation
        self.regiontree = construct_tree(parcellation.regions,
                rootname=NAME2IDENTIFIER(parcellation.name))

    def get_maps(self, space_key):
        """
        Get the volumetric maps for the selected parcellation in the requested
        template space. Note that this sometimes included multiple Nifti
        objects. For example, the Julich-Brain atlas provides two separate
        maps, one per hemisphere.

        Parameters
        ----------
        space_key : template space key

        Yields
        ------
        A dictionary of nibabel Nifti objects representing the volumetric map.
        The key of each object is a string that indicates which part of the
        brain each map describes, and may be used to identify the proper
        region name. In case of Julich-Brain, for example, it is "left
        hemisphere" and "right hemisphere".
        """
        space_id = key2id(space_key)
        if space_id not in self.__parcellation__.maps.keys():
            logging.error('The selected atlas parcellation is not available in the requested space.')
            logging.error('    Selected parcellation: {}'.format(self.__parcellation__.name))
            logging.error('    Requested space:       {}'.format(space_key))
            return None
        print('Loading 3D map for space ', space_key)
        mapurl = self.__parcellation__.maps[space_id]

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

    def get_mask(self, space_key):
        """
        Returns a binary mask  in the given space, where nonzero values denote
        voxels corresponding to the current region selection of the atlas. 

        WARNING: Note that for selections of subtrees of the region hierarchy, this
        might include holes if the leaf regions are not completly covering
        their parent and the parent itself has no label index in the map.

        Parameters
        ----------
        space : str
            Template space key
        """
        # remember that some parcellations are defined with multiple / split maps
        print(space_key)
        return self._get_regionmask(space_key,self.selection)

    @lru_cache(maxsize=5)
    def _get_regionmask(self,space_key,regiontree):
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
        space_key : str
            Template space key
        regiontree : Region
            A region from the region hierarchy (could be any of the root, a
            subtree, or a leaf)
        """
        print("Computing the mask for {} in {}".format(
            regiontree.name, space_key))
        maps = self.get_maps(space_key)
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

        TODO model the MNI URLs in the space ontology

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        if space['@id'] not in self._spaces:
            logging.error('The selected atlas does not support the requested reference space.')
            logging.error('    Requested space:       {}'.format(space['name']))
            return None

        print('Loading template image for space',space['name'])
        if 'templateFile' in space.keys():
            filename = download_file( space['templateUrl'], 
                    ziptarget=space["templateFile"])
        else:
            filename = download_file( space['templateUrl'])
        if filename is not None:
            return nib.load(filename)
        else:
            return None

    def select_region(self,region_id):
        """
        Selects a particular region. 

        TODO test carefully for selections of branching points in the region
        hierarchy, then managing all regions under the tree. This is nontrivial
        because for incomplete parcellations, the union of all child regions
        might not represent the complete parent node in the hierarchy.

        Parameters
        ----------
        region_id : str
            Id of the region to be selected, which is its full name converted
            by brainscapes' NAME2IDENTIFIER function.

        Yields
        ------
        True, if selection was successful, otherwise False.
        """
        regions_by_id = {NAME2IDENTIFIER(r.name):r for r in self.regiontree.descendants}
        if region_id in regions_by_id.keys():
            self.selection = regions_by_id[region_id]
            logging.info('Selected region {}'.format(self.selection.name))
            return True
        else:
            return False

    def inside_selection(self,space_key,position):
        """
        Verifies wether a position in the given space is inside the current
        selection.
        """
        space_id = key2id(space_key)
        assert(space_id in self._spaces)
        # NOTE since get_mask is lru-cached, this is not necessary slow
        mask = self.get_mask(space_key)
        if np.any(np.array(position)>=mask.dataobj.shape):
            return False
        if mask[position[0],position[1],position[2]]==0:
            return False
        return True

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


if __name__ == '__main__':
    atlas = Atlas()

    # atlas.get_maps('mySpace')
    # atlas.get_template("template")
    atlas.regiontree.print_hierarchy()
    print('*******************************')
    print(atlas.regiontree.find('hOc1'))
    print('*******************************')
    print(atlas.regiontree.find('LB (Amygdala) - left hemisphere'))
    print('******************************')
    print(atlas.regiontree.find('Ch 123 (Basal Forebrain) - left hemisphere'))
