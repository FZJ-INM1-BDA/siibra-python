from pathlib import Path
from zipfile import ZipFile
import os

import requests
import json
import nibabel as nib

from brainscapes.parcellations import Parcellations
from brainscapes.pmap_service import retrieve_probability_map
from brainscapes.region import Region


class Atlas:

    schema = Parcellations().CYTOARCHITECTONIC_MAPS
    _tmp_directory = 'brainscapes_tmp'
    _allowed_templates = [
        'mni_icbm152_t1_tal_nlin_asym_09c.nii',
        'colin27_t1_tal_lin.nii'
    ]

    def __init__(self):
        self._create_tmp_dir()

    def _create_tmp_dir(self):
        if not os.path.exists(self._tmp_directory):
            os.mkdir(self._tmp_directory)

    def select_parcellation_schema(self, schema):
        """
        Select a parcellation to work with.
        The default value is: Cytoarchitectonic maps

        :param schema:
        """
        self.schema = schema

    def get_map(self, space):
        """
        Getting a map (as nifti) for selected schema and given space.
        Map files are downloaded from cscs objectstore once and will be cached for further usage
        :param space:
        :return: nibabel image
        """
        print('getting map for: ' + space['id'])
        for sp in self.schema['availableIn']:
            if sp['@id'] == space['id']:
                url = sp['mapUrl']
                req = requests.get(url)
                if req is not None and req.status_code == 200:
                    filename = self._tmp_directory + '/' + req.headers['X-Object-Meta-Orig-Filename']
                    if not os.path.exists(filename):
                        with open(filename, 'wb') as code:
                            code.write(req.content)
                    return nib.load(filename)
        # throw error TODO
        '''
        - error on response status != 200
        - error on file read
        - Nibable error
        - handle error, when no filename header is set
        - error or None when space not known
        - unexpected error
        '''

    def get_template(self, space, resolution_mu=0, roi=None):
        """
        Getting a template (as nifti) for selected schema and space.
        Template files are downloade from www.bic.mni.mcgill.ca once and will be cached for further usage
        :param space:
        :param resolution_mu:
        :param roi:
        :return: nibabel image
        """
        print('getting template for: ' + space['id'] + ', with resolution: ' + str(resolution_mu))
        for sp in self.schema['availableIn']:
            if sp['@id'] == space['id']:
                # do request only, if file not yet downloaded
                download_filename = self._tmp_directory + '/' + space['shortName']
                if not os.path.exists(download_filename):
                    print('downloading a big file, this could take some time')
                    url = space['templateUrl']
                    req = requests.get(url)
                    if req is not None and req.status_code == 200:
                        # Write temporary zip file
                        with open(download_filename, 'wb') as code:
                            code.write(req.content)
                # Extract temporary zip file
                with ZipFile(download_filename, 'r') as zip_ref:
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename[-1] == '/':
                            continue
                        zip_info.filename = os.path.basename(zip_info.filename)
                        if zip_info.filename in self._allowed_templates:
                            zip_ref.extract(zip_info, self._tmp_directory)
                            return nib.load(self._tmp_directory + '/' + zip_info.filename)
        # throw error
        '''
        - error on response status != 200
        - error on file read
        - error on zipfile functions
        - Nibable error
        - error or None when space not known
        - unexpected error
        '''

    def get_region(self, region):
        print('getting region: ' + region)
        regions = self.regions()
        return self._check_for_region(region, regions)

    def _check_for_region(self, region, regions):
        for reg in regions:
            if reg['name'] == region:
                return {
                    'name': reg['name'],
                    'rgb': reg['rgb'],
                    # 'position': reg['position']
                }
            else:
                if len(reg['children']) != 0:
                    data = self._check_for_region(region, reg['children'])
                    if data:
                        return data
                else:
                    if reg['name'] == region:
                        return {
                            'name': reg['name'],
                            'rgb': reg['rgb'],
                            # 'position': reg['position']
                        }
        return None

    def regions(self):
        print('getting all regions')
        filename = self.schema['shortName'] + '.json'
        path = Path(__file__).parent / '../definitions/parcellations/' / filename
        with open(path, 'r') as jsonfile:
            data = json.load(jsonfile)
        return data['regions']

    def connectivity_sources(self):
        print('getting connectivity sources')
        return retrieve_probability_map(Region('Area-Fp1', 'colin', 'par1'), 'left', 0.2)

    def connectivity_matrix(self, src=None):
        print('get connectivity matrix for src: ' + src)

    def connectivity_filter(self, src=None):
        print('connectivity filter for src: ' + src)


if __name__ == '__main__':
    atlas = Atlas()

    # atlas.get_map('mySpace')
    # atlas.get_template("template")
    # data = atlas.regions()
    # print(data)
    print('*******************************')
    print(atlas.get_region('LB (Amygdala) - left hemisphere'))
    print('******************************')
    print(atlas.get_region('Ch 123 (Basal Forebrain) - left hemisphere'))
