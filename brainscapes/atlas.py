from zipfile import ZipFile
import os
from pkg_resources import resource_filename
import logging

import json
import nibabel as nib

from .pmap_service import retrieve_probability_map
from .region import Region
from .ontologies import atlases,parcellations,spaces
from .retrieval import download_file

class Atlas:

    # directory for cached files
    _tmp_directory = 'brainscapes_tmp'
    # templates that should be used from www.bic.mni.mcgill.ca
    # TODO i would put this information with the space ontologies. We have to
    # extend the concept of a simple URL and allow to give a URL to the zip and
    # the target filename. We should also chekc wether we are free to redistribute.
    _allowed_templates = [
        'mni_icbm152_t1_tal_nlin_asym_09c.nii',
        'colin27_t1_tal_lin.nii'
    ]
    __atlas__ = atlases.MULTILEVEL_HUMAN_ATLAS
    __parcellation__ =  parcellations.CYTOARCHITECTONIC_MAPS

    def __init__(self):
        # FIXME uses Python's temp directory functions for amore platform
        # independent solution
        if not os.path.exists(self._tmp_directory):
            os.mkdir(self._tmp_directory)

    def select_parcellation_scheme(self,parcellation):
        """
        Select a different parcellation for the atlas.

        :param schema:
        """
        # TODO need more robust handling with ontology definition schemes, they should become well defined objects
        assert('@id' in parcellation.keys())
        if parcellation['@id'] not in self.__atlas__['parcellations']:
            logging.error('The requested parcellation is not supported by the selected atlas.')
            logging.error('    Parcellation:  '+parcellation['name'])
            logging.error('    Atlas:         '+self.__atlas__['name'])
            raise Exception('Invalid Parcellation')
        self.__parcellation__ = parcellation

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
        filename = download_file(self.__parcellation__['maps'][space['@id']],self._tmp_directory)
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
        if space['@id'] not in self.__atlas__['spaces'].keys():
            logging.error('The selected atlas does not support the requested reference space.')
            logging.error('    Requested space:       {}'.format(space['name']))
            return None

        filename = download_file(space['url'],self._tmp_directory)

        # Extract temporary zip file
        with ZipFile(filename, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == '/':
                    continue
                zip_info.filename = os.path.basename(zip_info.filename)
                if zip_info.filename in self._allowed_templates:
                    zip_ref.extract(zip_info, self._tmp_directory)
                    return nib.load(self._tmp_directory + '/' + zip_info.filename)

        # not successful
        return None

    def get_region(self, region):
        regions = self.regions()
        return self._check_for_region(region, regions)

    def _check_for_region(self, region, regions):
        for reg in regions:
            if reg['name'] == region:
                return Region(reg['name'], self.schema)
                # return {
                #     'name': reg['name'],
                #     'rgb': reg['rgb'],
                #     # 'position': reg['position']
                # }
            else:
                if len(reg['children']) != 0:
                    data = self._check_for_region(region, reg['children'])
                    if data:
                        return data
                else:
                    if reg['name'] == region:
                        return Region(reg['name'], self.schema)
                        # {
                        #     'name': reg['name'],
                        #     'rgb': reg['rgb'],
                        #     # 'position': reg['position']
                        # }
        return None

    def regions(self):
        with open(resource_filename( 
            'brainscapes.ontologies.parcellations',
            self.schema['shortName'] + '.json'), 'r') as jsonfile:
            data = json.load(jsonfile)
        return data['regions']

    def connectivity_sources(self):
        print('getting connectivity sources')
        #TODO parameterize getting the probability map
        return retrieve_probability_map(Region('Area-Fp1', 'colin', 'par1'), 'left', 0.2)

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
    data = atlas.regions()
    print(data)
    print('*******************************')
    print(atlas.get_region('LB (Amygdala) - left hemisphere'))
    print('******************************')
    print(atlas.get_region('Ch 123 (Basal Forebrain) - left hemisphere'))
