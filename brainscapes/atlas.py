import logging
import nibabel as nib
import numpy as np
import json
from collections import defaultdict

from . import NAME2IDENTIFIER
from .region import construct_tree
from .ontologies import atlases, parcellations, spaces
from .retrieval import download_file

class Atlas:

    def __init__(self,definition=atlases.MULTILEVEL_HUMAN_ATLAS):
        # Set an atlas from a json definition. As a default, multilevel human
        # atlas definition is used. The first parcellation in the atlas
        # definition is selected as the default parcellation.
        self._atlas = definition
        self.regiontree = None
        self.features = defaultdict(list)
        self.select_parcellation(
                parcellations[definition['parcellations'][0]])
        self.selection = self.regiontree

    def select_parcellation(self, parcellation):
        """
        Select a different parcellation for the atlas.

        :param schema:
        """
        # TODO need more explicit formalization and testing of ontology
        # definition schemes
        assert('@id' in parcellation.keys())
        if parcellation['@id'] not in self._atlas['parcellations']:
            logging.error('The requested parcellation is not supported by the selected atlas.')
            logging.error('    Parcellation:  '+parcellation['name'])
            logging.error('    Atlas:         '+self._atlas['name'])
            logging.error(parcellation['@id'],self._atlas['parcellations'])
            raise Exception('Invalid Parcellation')
        self.__parcellation__ = parcellation
        self.regiontree = construct_tree(parcellation['regions'],
                rootname=NAME2IDENTIFIER(parcellation['name']))

    def get_maps(self, space):
        """
        Get the volumetric maps for the selected parcellation in the requested
        template space. Note that this sometimes included multiple Nifti
        objects. For example, the Julich-Brain atlas provides two separate
        maps, one per hemisphere.

        Parameters
        ----------
        space : template space definition, given as a dictionary with an '@id' key

        Yields
        ------
        A dictionary of nibabel Nifti objects representing the volumetric map.
        The key of each object is a string that indicates which part of the
        brain each map describes, and may be used to identify the proper
        region name. In case of Julich-Brain, for example, it is "left
        hemisphere" and "right hemisphere".
        """
        if space['@id'] not in self.__parcellation__['maps'].keys():
            logging.error('The selected atlas parcellation is not available in the requested space.')
            logging.error('    Selected parcellation: {}'.format(self.__parcellation__['name']))
            logging.error('    Requested space:       {}'.format(space))
            return None
        print('Loading 3D map for space ', space['name'])
        mapurl = self.__parcellation__['maps'][space['@id']]

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

    def get_mask(self, space):
        """
        Returns a binary mask  in the given space, where nonzero values denote
        voxels corresponding to the current region selection. 

        WARNING: Note that for selections of subtrees of the region hierarchy, this
        might include holes if the leaf regions are not completly covering
        their parent and the parent itself has no label index in the map.

        Parameters
        ----------
        space : str
            Template space definition, given as a dictionary with an '@id' key
        """
        # remember that some parcellations are defined with multiple / split maps
        maps = self.get_maps(space)
        mask = affine = header = None 
        for mlabel,m in maps.items():
            D = np.array(m.dataobj)
            if mask is None: 
                # copy metadata for output mask from the first map!
                mask = np.zeros_like(D)
                affine = m.affine
                header = m.header
            for region in self.selection.iterate():
                if 'labelIndex' not in region.attrs.keys():
                    continue
                if region.attrs['labelIndex'] is None:
                    continue
                mask[D==int(region.attrs['labelIndex'])]=1

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
        if space['@id'] not in self._atlas['spaces']:
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
