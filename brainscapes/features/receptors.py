import io
import logging

import pandas as pd
import PIL.Image as Image
from os import path
from collections import defaultdict

from brainscapes import kg_service, retrieval
from brainscapes.authentication import Authentication
from brainscapes.features.feature import RegionalFeature,FeaturePool




class termcolor:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

class ReceptorDistribution(RegionalFeature):

    profiles = {}
    autoradiographs = {}
    autoradiographs_files = {}
    receptors = defaultdict(dict)

    def __init__(self, region, kg_response):

        RegionalFeature.__init__(self,region)

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
                        self.receptors[receptor_type]['profile'] = pd.read_csv(bytestream, sep='\t')
                else:
                    logging.debug('Expected .tsv for profile, got {}: {}'.format(suffix, fname))

            # Receive autoradiographs, if any
            if '_ar_' in fname:
                receptor_type, basename = fname.split("/")[-2:]
                if receptor_type in basename:
                    bytestream = self._get_bytestream_from_file(fname)
                    self.receptors[receptor_type]['autoradiograph'] = Image.open(bytestream)
                    self.receptors[receptor_type]['autoradiograph_files'] = fname

            # receive fingerprint, if any
            if '_fp_' in fname:
                bytestream = self._get_bytestream_from_file(fname)
                self.fingerprint = pd.read_csv(bytestream, sep='\t')

    def _get_bytestream_from_file(self, fname):
        file = retrieval.download_file(fname)
        with open(file, 'rb') as f:
            return io.BytesIO(f.read())

    def __str__(self):
        return "\n".join(
                [termcolor.BOLD+"{!s:20} {!s:>10} {!s:>20}".format('Type','profile','autoradiograph')+termcolor.END] +
                ["{!s:20} {!s:>10} {!s:>20}".format(k,'profile' in D.keys(),'autoradiograph' in D.keys())
                    for k,D in self.receptors.items()])


class ReceptorQuery(FeaturePool):

    _FEATURETYPE = ReceptorDistribution

    def __init__(self):

        FeaturePool.__init__(self)
        kg_query = kg_service.execute_query_by_id('minds', 'core', 'dataset', 'v1.0.0', 'bs_datasets_tmp')
        for kg_result in kg_query['results']:
            region_names = [e['https://schema.hbp.eu/myQuery/name'] 
                    for e in kg_result['https://schema.hbp.eu/myQuery/parcellationRegion']]
            for region_name in region_names:
                self.register(ReceptorDistribution(region_name,kg_result))

if __name__ == '__main__':
    auth = Authentication.instance()
    auth.set_token('eyJhbGciOiJSUzI1NiIsImtpZCI6ImJicC1vaWRjIn0.eyJleHAiOjE2MDA4NTgyMjAsInN1YiI6IjMwODExMCIsImF1ZCI6WyJuZXh1cy1rZy1zZWFyY2giXSwiaXNzIjoiaHR0cHM6XC9cL3NlcnZpY2VzLmh1bWFuYnJhaW5wcm9qZWN0LmV1XC9vaWRjXC8iLCJqdGkiOiIzMWYyYTUzNS04NjI0LTQ3NDQtYmEzNS00NDI4NWFlMTI3YWIiLCJpYXQiOjE2MDA4NDM4MjAsImhicF9rZXkiOiJmZmJhYzViYmYyNzdmYzc3NDFhYTBiYjcxNGQxZDAzMWVjNmQxOGNlIn0.b1J8i524HKE2d97VpG9oJppV04TfWjtPCOhN-QPGyH5TG_KfuBZvMeDuUs25umbxmJuSnisdPny-JXJlnbBg7OGvYFeCP1KP3ywmGxSOeoMYx5hhJ41GIwyoXjKhKv6rVQpyB21Y0-MwX2sg1iJz_un7P6kTo1OdMVfX_ulL_v4')
    pool = ReceptorQuery()
    print(pool)
