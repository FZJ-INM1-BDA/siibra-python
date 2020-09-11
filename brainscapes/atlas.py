import os
import logging
import json
import nibabel as nib
from anytree import Node, RenderTree

from .region import Region
from .ontologies import atlases, parcellations, spaces
from .retrieval import download_file, get_file_from_zip

def regionlist(regions,parent=None):
    """
    From a dictionary of regions, as given by brainscapes parcellation
    definitions, build a consolidated hierarchy of names.
    """
    trees = []
    for regiondef in regions:
        newnode = Node(regiondef['name'],parent=parent)
        if 'children' in regiondef.keys():
            _ = regionlist(regiondef['children'],parent=newnode)
        trees.append(newnode)
    return trees

def search_region(regions,name,exact=True):
    """
    In a dictionary of regions, as given by brainscapes parcellation
    definitions, perform a recursive tree search for a given region name.

    If extact==False, will return all regions that match name as a substring.
    """
    matches = []
    for regiondef in regions:

        if exact and (regiondef['name']==name):
            return Region(regiondef)

        if (not exact) and (name in regiondef['name']):
            matches.append(Region(regiondef))

        if 'children' in regiondef.keys():
            more_matches = search_region(regiondef['children'],name,exact=exact)
            matches += more_matches

    # TODO should rather be a special "Undefined Region"
    return None if exact else matches
        

class Atlas:

    # directory for cached files
    _tmp_directory = 'brainscapes_tmp'
    # templates that should be used from www.bic.mni.mcgill.ca
    # TODO I would put this information with the space ontologies. We have to
    # extend the concept of a simple URL and allow to give a URL to the zip and
    # the target filename. We should also chekc wether we are free to redistribute.
    _allowed_templates = [
        'mni_icbm152_t1_tal_nlin_asym_09c.nii',
        'colin27_t1_tal_lin.nii'
    ]

    def __init__(self):
        # FIXME uses Python's temp directory functions for amore platform
        # independent solution
        if not os.path.exists(self._tmp_directory):
            os.mkdir(self._tmp_directory)
        self.__atlas__ = atlases.MULTILEVEL_HUMAN_ATLAS
        self.__regions__ = []
        self.select_parcellation_scheme(parcellations.CYTOARCHITECTONIC_MAPS)

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
        self.__regions__ = parcellation['regions']

    def get_map(self, space):
        """
        Get a volumetric map for the selected parcellation in the requested
        template space, if available.

        Parameters
        ----------
        space : template space definition, given as a dictionary with an '@id' key

        Yields
        ------
        A nibabel Nifti object representing the volumetric map, or None if not available.
        TODO Returning None is not ideal, requires to implement a test on the other side. 
        """
        print('Retrieving map for ' + space['name'])
        if space['@id'] not in self.__parcellation__['maps'].keys():
            logging.error('The selected atlas parcellation is not available in the requested space.')
            logging.error('    Selected parcellation: {}'.format(self.__parcellation__['name']))
            logging.error('    Requested space:       {}'.format(space))
            return None
        filename = download_file(self.__parcellation__['maps'][space['@id']], self._tmp_directory)
        if filename is not None:
            return nib.load(filename)
        else:
            return None

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
        print('Retrieving template for ' + space['name'] + ', with resolution: ' + str(resolution_mu))
        if space['@id'] not in self.__atlas__['spaces']:
            logging.error('The selected atlas does not support the requested reference space.')
            logging.error('    Requested space:       {}'.format(space['name']))
            return None

        filename = download_file(space['templateUrl'], self._tmp_directory)
        return get_file_from_zip(filename)

    def get_region(self, regionname):
        return search_region(self.__regions__, regionname, exact=True)

    def search_region(self, substring):
        return search_region(self.__regions__, substring, exact=False)

    def regionhierarchy(self):
        """
        Prints a hierarchy of defined region names.
        """
        for tree in regionlist(self.__regions__):
            for pre, _, node in RenderTree(tree):
                print("%s%s" % (pre, node.name))

    def connectivity_sources(self):
        print('getting connectivity sources')
        #TODO Implement

    def connectivity_matrix(self, src=None):
        print('get connectivity matrix for src: ' + src)
        #TODO implement

    def connectivity_filter(self, src=None):
        print('connectivity filter for src: ' + src)
        #TODO implement


if __name__ == '__main__':
    atlas = Atlas()

    # atlas.get_map('mySpace')
    # atlas.get_template("template")
    atlas.regionhierarchy()
    print('*******************************')
    print(atlas.search_region('hOc1'))
    print('*******************************')
    print(atlas.get_region('LB (Amygdala) - left hemisphere'))
    print('******************************')
    print(atlas.get_region('Ch 123 (Basal Forebrain) - left hemisphere'))
