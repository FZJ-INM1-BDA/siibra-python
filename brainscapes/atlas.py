import logging
import nibabel as nib
import numpy as np
from tempfile import mkdtemp
import json
from collections import defaultdict

from .region import Region,construct_tree
from .ontologies import atlases, parcellations, spaces
from .features import sources as featuresources
from .retrieval import download_file

class Atlas:

    def __init__(self,cachedir=None):
        self._cachedir = mkdtemp() if cachedir is None else cachedir
        self.__atlas__ = atlases.MULTILEVEL_HUMAN_ATLAS
        self.regiontree = None
        self.features = defaultdict(list)
        self.select_parcellation_scheme(parcellations.JULICH_BRAIN_PROBABILISTIC_CYTOARCHITECTONIC_ATLAS)

    def select_parcellation_scheme(self, parcellation):
        """
        Select a different parcellation for the atlas.

        :param schema:
        """
        # TODO need more explicit formalization and testing of ontology
        # definition schemes
        assert('@id' in parcellation.keys())
        if parcellation['@id'] not in self.__atlas__['parcellations']:
            logging.error('The requested parcellation is not supported by the selected atlas.')
            logging.error('    Parcellation:  '+parcellation['name'])
            logging.error('    Atlas:         '+self.__atlas__['name'])
            raise Exception('Invalid Parcellation')
        self.__parcellation__ = parcellation
        self.regiontree = construct_tree(parcellation['regions'])

        # load features
        # TODO refactor
        for targetname,url in featuresources.items():
            filename = download_file(url, self._cachedir, targetname=targetname)
            with open(filename,'r') as f:
                for item in json.load(f):
                    if item['parcellation']==self.__parcellation__['name']:
                        self.features[item['type']].append(item)
                        print("Feature loaded:",item['type'],"/",self.features[item['type']][-1]['name'])

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
                filename = download_file(url, self._cachedir)
                if filename is not None:
                    maps[label] = nib.load(filename)
        else:
            filename = download_file(mapurl, self._cachedir)
            maps[''] = nib.load(filename)
        
        return maps

    def get_template(self, space, resolution_mu=0, roi=None):
        """
        Get the volumetric reference template image for the given space.

        Parameters
        ----------
        space : template space definition, given as a dictionary with an '@id' key
        resolution : Desired target pixel spacing in micrometer (default: native spacing)
        roi : 3D region of interest (not yet implemented)

        TODO model the MNI URLs in the space ontology

        Yields
        ------
        A nibabel Nifti object representing the reference template, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        if space['@id'] not in self.__atlas__['spaces']:
            logging.error('The selected atlas does not support the requested reference space.')
            logging.error('    Requested space:       {}'.format(space['name']))
            return None

        print('Loading template image for space',space['name'])
        if 'templateFile' in space.keys():
            filename = download_file( space['templateUrl'], self._cachedir, 
                    ziptarget=space["templateFile"])
        else:
            filename = download_file( space['templateUrl'], self._cachedir)
        if filename is not None:
            return nib.load(filename)
        else:
            return None

    def regions(self):
        return [n
                for n in self.regiontree.descendants 
                if n.is_leaf]

    def connectivity_sources(self):
        return [f['name'] for f in self.features['Connectivity Profiles']]
        #TODO Implement

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
