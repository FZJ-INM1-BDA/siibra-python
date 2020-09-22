import io
import logging

import pandas as pd
import PIL.Image as Image
from os import path

from brainscapes import kg_service, retrieval
from brainscapes.authentication import Authentication
from brainscapes.features.feature import Feature


class ReceptorDistributionFeature(Feature):

    # data rows
    profiles = {}
    # images
    autoradiographs = {}
    # image urls
    autoradiographs_files = {}

    def __init__(self, kg_response):

        self.regions = [e['https://schema.hbp.eu/myQuery/name']
                        for e in kg_response['https://schema.hbp.eu/myQuery/parcellationRegion']]

        for fname in kg_response['https://schema.hbp.eu/myQuery/v1.0.0']:

            if 'receptors.tsv' in fname:
                bytestream = self._get_bytestream_from_file(fname)
                self.__symbols = pd.read_csv(bytestream, sep='\t')
                self.receptor_label = {r._1: r._2
                                       for r in self.__symbols.itertuples()}

            # Receive cortical profiles, if any
            if '_pr_' in fname:
                suffix = path.splitext(fname)[-1]
                if suffix == '.tsv':
                    receptor_type, basename = fname.split("/")[-2:]
                    if receptor_type in basename:
                        bytestream = self._get_bytestream_from_file(fname)
                        self.profiles[receptor_type] = pd.read_csv(bytestream, sep='\t')
                else:
                    logging.debug('Expected .tsv for profile, got {}: {}'.format(suffix, fname))

            if '_ar_' in fname:
                receptor_type, basename = fname.split("/")[-2:]
                if receptor_type in basename:
                    bytestream = self._get_bytestream_from_file(fname)
                    self.autoradiographs[receptor_type] = Image.open(bytestream)
                    self.autoradiographs_files [receptor_type] = fname

            if '_fp_' in fname:
                bytestream = self._get_bytestream_from_file(fname)
                self.fingerprint = pd.read_csv(bytestream, sep='\t')

    def matches_selection(self, atlas):
        return True

    def _get_bytestream_from_file(self, fname):
        file = retrieval.download_file(fname)
        with open(file, 'rb') as f:
            return io.BytesIO(f.read())


_receptor_data_repo = {}


def get_receptor_data_by_region(region_name):
    if region_name not in _receptor_data_repo.keys():
        kg_result = kg_service.execute_query_by_id('minds', 'core', 'dataset', 'v1.0.0', 'bs_datasets_tmp', params='&regionname=' + region_name)
        for region in kg_result['results']:
            region_names = [e['https://schema.hbp.eu/myQuery/name'] for e in region['https://schema.hbp.eu/myQuery/parcellationRegion']]
            for r in region_names:
                _receptor_data_repo[r] = ReceptorDistributionFeature(region)
    return _receptor_data_repo[region_name]


if __name__ == '__main__':
    auth = Authentication.instance()
    auth.set_token('eyJhbGciOiJSUzI1NiIsImtpZCI6ImJicC1vaWRjIn0.eyJleHAiOjE2MDA3NzYyOTAsInN1YiI6IjMwODExMCIsImF1ZCI6WyJuZXh1cy1rZy1zZWFyY2giXSwiaXNzIjoiaHR0cHM6XC9cL3NlcnZpY2VzLmh1bWFuYnJhaW5wcm9qZWN0LmV1XC9vaWRjXC8iLCJqdGkiOiI2ZWU2N2MyMC00MjNmLTRmYzgtYjA1ZC02MWI3OThjMjRhNjIiLCJpYXQiOjE2MDA3NjE4OTAsImhicF9rZXkiOiJiN2QxMWZmZTdiZmZmZDhmNTE4YWUzNzM3ZjA1ODk4OTRjMTAzMzRlIn0.CRqAOyO1sfPb-rH8w9qG0_-t9n5adbnoyr8Z2LYws7pBRQbWbcT3-SyXc2TnM0x3tfnl0N2w2tBs8-CZEf77pbb5xijeZJKI8PnNonyLu3ALELD-B0luWgz666fy4_fTVertF7gxU98pZQG7bccVmJE7CvPA3Teie5D3-LCaNLE')
    print(get_receptor_data_by_region('Area 4p (PreCG)'))
    print(get_receptor_data_by_region('Area 4p (PreCG)').profiles)
    print(get_receptor_data_by_region('Area 4p (PreCG)').autoradiographs)
    print(get_receptor_data_by_region('Area 4p (PreCG)').fingerprint)
    # print(__receptor_data_repo)
