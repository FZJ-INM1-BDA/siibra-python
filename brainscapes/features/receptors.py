import io
import logging

import pandas as pd
import PIL.Image as Image
from os import path

from brainscapes import kg_service, retrieval
from brainscapes.features.feature import RegionalFeature,FeaturePool


class ReceptorDistribution(RegionalFeature):

    profiles = {}
    autoradiographs = {}

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
                        self.profiles[receptor_type] = pd.read_csv(bytestream, sep='\t')
                else:
                    logging.debug('Expected .tsv for profile, got {}: {}'.format(suffix, fname))

            if '_ar_' in fname:
                receptor_type, basename = fname.split("/")[-2:]
                if receptor_type in basename:
                    bytestream = self._get_bytestream_from_file(fname)
                    self.autoradiographs[receptor_type] = Image.open(bytestream)

            if '_fp_' in fname:
                bytestream = self._get_bytestream_from_file(fname)
                self.fingerprint = pd.read_csv(bytestream, sep='\t')

    def _get_bytestream_from_file(self, fname):
        file = retrieval.download_file(fname)
        with open(file, 'rb') as f:
            return io.BytesIO(f.read())

    def __str__(self):
        return "\n".join([
            "Receptors in area '{}':".format(self.region),
            "Profiles: {}".format(",".join(self.profiles.keys())),
            "Thumbnails: {}".format(",".join(self.autoradiographs.keys()))
            ])


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
    pool = ReceptorQuery()
    print(pool)
