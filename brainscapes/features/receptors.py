import io
import PIL.Image as Image
from os import path

from brainscapes import kg_service, retrieval, logger
from brainscapes.authentication import Authentication
from brainscapes.features.feature import RegionalFeature,FeatureExtractor
from brainscapes.termplot import FontStyles as style

def get_bytestream_from_file(url):
    fname = retrieval.download_file(url)
    with open(fname, 'rb') as f:
        return io.BytesIO(f.read())

def decode_tsv(url):
    bytestream = get_bytestream_from_file(url)
    header = bytestream.readline()
    lines = bytestream.readlines() 
    sep = b'{' if b'{' in lines[0] else b'\t' 
    keys = [n.decode('utf8') for n in header.split(sep)] 
    return  { l.split(sep)[0].decode('utf8') : dict(
        zip(keys[1:], [v.decode('utf8') for v in l.split(sep)[1:]])) 
        for l in lines }

def edits1(word):
    """All edits that are one edit away from `word`.

    From Peter Norvig, see http://norvig.com/spell-correct.html.
    """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


class ReceptorDistribution(RegionalFeature):


    def __init__(self, region, file_urls):

        RegionalFeature.__init__(self,region)
        self.profiles = {}
        self.autoradiographs = {}
        self.fingerprint = None
        self.symbols = {}

        # find symbols first
        for url in file_urls:

            if 'receptors.tsv' in url:
                self.symbols = decode_tsv(url)

        for url in file_urls:

            # Receive cortical profiles, if any
            if '_pr_' in url:
                suffix = path.splitext(url)[-1]
                if suffix == '.tsv':
                    rtype, basename = url.split("/")[-2:]
                    if rtype in basename:
                        data = decode_tsv(url)
                        # column headers are sometimes messed up, so we fix the 2nd value
                        densities = {int(k):float(list(v.values())[1]) 
                                for k,v in data.items()
                                if k.isnumeric()}
                        rtype = self._check_rtype(rtype)
                        self.profiles[rtype] = densities
                else:
                    logger.debug('Expected .tsv for profile, got {}: {}'.format(suffix, url))

            # Receive autoradiographs, if any
            if '_ar_' in url:
                rtype, basename = url.split("/")[-2:]
                if rtype in basename:
                    bytestream = get_bytestream_from_file(url)
                    rtype = self._check_rtype(rtype)
                    self.autoradiographs[rtype] = Image.open(bytestream)

            # receive fingerprint, if any
            if '_fp_' in url:
                self.fingerprint = decode_tsv(url)


    def _check_rtype(self,rtype):
        """ 
        Verify that the receptor type name matches the symbol table. 
        Return if ok, fix if a close match is found, raise Excpetion otherwise.
        """
        if rtype in self.symbols.keys():
            return rtype
        # fix if only different by 1 letter from a close match in the symbol table
        close_matches = list(edits1(rtype).intersection(self.symbols.keys()))
        if len(close_matches)==1:
            prev,new = rtype, close_matches[0]
            logger.warn("Receptor type identifier '{}' replaced by '{}' for {}".format(
                prev,new, self.region))
            return new
        else:
            raise ValueError("Inconsistent rtype '{}' in {}".format(
                rtype, self.region))
        

    def __str__(self):
        """ TODO improve """
        return "\n"+"\n".join(
                [style.BOLD+"Receptor density measurements for area {}".format(self.region)+style.END] +
                [style.ITALIC+"{!s:20} {!s:>10} {!s:>20}".format('Type','profile','autoradiograph')+style.END] +
                ["{!s:20} {!s:>10} {!s:>20}".format(k,'profile' in D.keys(),'autoradiograph' in D.keys())
                    for k,D in self.profiles.items()])


class ReceptorQuery(FeatureExtractor):

    _FEATURETYPE = ReceptorDistribution

    def __init__(self):

        FeatureExtractor.__init__(self)
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
    extractor = ReceptorQuery()
    print(extractor)
