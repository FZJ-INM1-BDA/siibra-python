import io
import pandas as pd
import PIL.Image as Image
from os import path
from collections import defaultdict

from brainscapes import kg_service, retrieval, logger
from brainscapes.authentication import Authentication
from brainscapes.features.feature import RegionalFeature,FeaturePool
from brainscapes.termplot import FontStyles as style

class ReceptorDistribution(RegionalFeature):


    def __init__(self, region, file_urls):

        RegionalFeature.__init__(self,region)
        self.receptors = defaultdict(dict)

        for url in file_urls:

            if 'receptors.tsv' in url:
                bytestream = self._get_bytestream_from_file(url)
                self.__symbols = pd.read_csv(bytestream, sep='\t')
                self.receptor_label = {r._1: r._2
                                       for r in self.__symbols.itertuples()}

            # Receive cortical profiles, if any
            if '_pr_' in url:
                suffix = path.splitext(url)[-1]
                if suffix == '.tsv':
                    receptor_type, basename = url.split("/")[-2:]
                    if receptor_type in basename:
                        bytestream = self._get_bytestream_from_file(url)
                        self.receptors[receptor_type]['profile'] = pd.read_csv(bytestream, sep='\t')
                else:
                    logger.debug('Expected .tsv for profile, got {}: {}'.format(suffix, url))

            # Receive autoradiographs, if any
            if '_ar_' in url:
                receptor_type, basename = url.split("/")[-2:]
                if receptor_type in basename:
                    bytestream = self._get_bytestream_from_file(url)
                    self.receptors[receptor_type]['autoradiograph'] = Image.open(bytestream)
                    self.receptors[receptor_type]['autoradiograph_files'] = url

            # receive fingerprint, if any
            if '_fp_' in url:
                bytestream = self._get_bytestream_from_file(url)
                self.fingerprint = pd.read_csv(bytestream, sep='\t')

    def _get_bytestream_from_file(self, url):
        file = retrieval.download_file(url)
        with open(file, 'rb') as f:
            return io.BytesIO(f.read())

    def __str__(self):
        return "\n"+"\n".join(
                [style.BOLD+"Receptor density measurements for area {}".format(self.region)+style.END] +
                [style.ITALIC+"{!s:20} {!s:>10} {!s:>20}".format('Type','profile','autoradiograph')+style.END] +
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
                file_urls = kg_result["https://schema.hbp.eu/myQuery/v1.0.0"]
                self.register(ReceptorDistribution(region_name,file_urls))


if __name__ == '__main__':
    auth = Authentication.instance()
    auth.set_token('eyJhbGci....')
    pool = ReceptorQuery()
    print(pool)
