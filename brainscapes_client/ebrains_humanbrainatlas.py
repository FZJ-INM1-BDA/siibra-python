import requests
import gzip
import zipfile
import json

from brainscapes_client.pmap_service import retrieve_probability_map
from brainscapes_client.region import Region


class Atlas:

    def say_hello(self):
        print("Hello World")

    def select_parcellation_schema(self, schema):
        self.schema = schema

    def get_map(self, space):
        print('getting map for: ' + space)
        url = 'https://object.cscs.ch/v1/AUTH_227176556f3c4bb38df9feea4b91200c/hbp-d000001_jubrain-cytoatlas_pub/18/MPM/mpmatlas_l_N10_nlin2icbm152casym_18_public_a5f6c95f2e7ff6f43b6bf7c816c37c8b.nii.gz'
        req = requests.get(url)

        if req is not None and req.status_code == 200:
            data_nii = gzip.decompress(req.content)
        filename = req.headers['X-Object-Meta-Orig-Filename']
        print(req.status_code)
        with open(filename, 'wb') as code:
            code.write(data_nii)

    def get_template(self, space, resolution_mu=0, roi=None):
        print('getting template for: ' + space + ', with resolution: ' + str(resolution_mu))
        # url = 'http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09c_nifti.zip'
        # req = requests.get(url)
        #
        # if req is not None and req.status_code == 200:
        #     data_nii = gzip.decompress(req.content)
        # filename = req.headers['X-Object-Meta-Orig-Filename']
        # print(req.status_code)
        # print(req.headers)
        # with open(filename, 'wb') as code:
        #     code.write(data_nii)

        zf = zipfile.ZipFile('../data/mni_icbm152_nlin_asym_09c_nifti.zip', 'r')
        # print(zf.namelist())
        return zf

    def get_region(self, region):
        print('getting region: ' + region)
        regions = self.regions()
        # region_data = next((reg for reg in regions if reg['name'] == region), None)

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
        with open('../data/parcellations/jubrain.json', 'r') as jsonfile:
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
    atlas.say_hello()

    # atlas.get_map('mySpace')
    # atlas.get_template("template")
    # data = atlas.regions()
    # print(data)
    print('*******************************')
    print(atlas.get_region('LB (Amygdala) - left hemisphere'))
    print('******************************')
    print(atlas.get_region('Ch 123 (Basal Forebrain) - left hemisphere'))
