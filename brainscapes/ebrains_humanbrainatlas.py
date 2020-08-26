from pathlib import Path

import requests
import json
import nibabel as nib

from brainscapes.parcellations import Parcellations
from brainscapes.pmap_service import retrieve_probability_map
from brainscapes.region import Region


class Atlas:

    schema = Parcellations().CYTOARCHITECTONIC_MAPS

    def select_parcellation_schema(self, schema):
        self.schema = schema

    def get_map(self, space):
        print('getting map for: ' + space['id'])
        for sp in self.schema['availableIn']:
            if sp['@id'] == space['id']:
                url = sp['mapUrl']
                req = requests.get(url)
                if req is not None and req.status_code == 200:
                    filename = req.headers['X-Object-Meta-Orig-Filename']
                    with open(filename, 'wb') as code:
                        code.write(req.content)
                    return nib.load(filename)
        # throw error

    def get_template(self, space, resolution_mu=0, roi=None):
        print('getting template for: ' + space['id'] + ', with resolution: ' + str(resolution_mu))
        print(space)
        for sp in self.schema['availableIn']:
            if sp['@id'] == space['id']:
                url = space['templateUrl']
                req = requests.get(url)
                print(req.headers)
                if req is not None and req.status_code == 200:
                    # data_nii = gzip.decompress(req.content)
                    filename = 'tmp-template.zip'#space['id']#req.headers['X-Object-Meta-Orig-Filename']#.replace('.zip', '')
                    with open(filename, 'wb') as code:
                        code.write(req.content)
                    return nib.load(filename)
        # throw error

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
