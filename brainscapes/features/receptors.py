import io
import PIL.Image as Image
from os import path

from .feature import RegionalFeature
from .extractor import FeatureExtractor
from ..authentication import Authentication
from ..termplot import FontStyles as style
from .. import ebrains, retrieval, logger

def get_bytestream_from_file(url):
    fname = retrieval.download_file(url)
    with open(fname, 'rb') as f:
        return io.BytesIO(f.read())

def unify_stringlist(L: list):
    """ Adds asterisks to strings that appear multiple times, so the resulting
    list has only unique strings but still the same length, order, and meaning. 
    For example: 
        unify_stringlist(['a','a','b','a','c']) -> ['a','a*','b','a**','c']
    """
    assert(all([isinstance(l,str) for l in L]))
    return [L[i]+"*"*L[:i].count(L[i]) for i in range(len(L))]

def edits1(word):
    """
    Produces a list of  all possible edits of a given word that can be produced
    by applying a single character modification.
    From Peter Norvig, see http://norvig.com/spell-correct.html.
    """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def decode_tsv(url):
    bytestream = get_bytestream_from_file(url)
    header = bytestream.readline()
    lines = [l.strip() 
            for l in bytestream.readlines() 
            if len(l.strip())>0]
    sep = b'{' if b'{' in lines[0] else b'\t' 
    keys = unify_stringlist([n.decode('utf8').strip() 
        for n in header.split(sep)])
    assert(len(keys)==len(set(keys)))
    return  { l.split(sep)[0].decode('utf8') : dict(
        zip(keys, [v.decode('utf8').strip() for v in l.split(sep)])) 
        for l in lines }


class ReceptorDistribution(RegionalFeature):

    def __init__(self, region, file_urls):

        RegionalFeature.__init__(self,region)
        self.profiles = {}
        self.autoradiographs = {}
        self.fingerprint = None
        self.profile_unit = None
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
                        units = {list(v.values())[3] for v in data.values()}
                        assert(len(units)==1)
                        self.profile_unit=next(iter(units))
                        # column headers are sometimes messed up, so we fix the 2nd value
                        densities = {int(k):float(list(v.values())[2]) 
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

    def __bool__(self):
        nonempty = any([ 
            len(self.profiles)>0,
            len(self.autoradiographs)>0,
            self.fingerprint is not None])
        return nonempty
    __nonzero__ = __bool__

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
        """ Outputs a small table of available profiles and autoradiographs. """
        if len(self.profiles)+len(self.autoradiographs)==0:
                return style.BOLD+"Receptor density for area {}".format(self.region)+style.END
        return "\n"+"\n".join(
                [style.BOLD+"Receptor densities for area {}".format(self.region)+style.END] +
                [style.ITALIC+"{!s:20} {!s:>8} {!s:>15} {!s:>11}".format(
                    'Type','profile','autoradiograph','fingerprint')+style.END] +
                ["{!s:20} {!s:>8} {!s:>15} {!s:>11}".format(
                    rtype,
                    'x'*(rtype in self.profiles),
                    'x'*(rtype in self.autoradiographs),
                    'x'*(rtype in self.fingerprint))
                    for rtype in self.symbols.keys()
                    if (rtype in self.profiles 
                        or rtype in self.autoradiographs)] )


class ReceptorQuery(FeatureExtractor):

    _FEATURETYPE = ReceptorDistribution

    def __init__(self):

        FeatureExtractor.__init__(self)
        kg_query = ebrains.execute_query_by_id('minds', 'core', 'dataset', 'v1.0.0', 'bs_datasets_tmp')
        for kg_result in kg_query['results']:
            region_names = [e['https://schema.hbp.eu/myQuery/name'] 
                    for e in kg_result['https://schema.hbp.eu/myQuery/parcellationRegion']]
            for region_name in region_names:
                file_urls = kg_result["https://schema.hbp.eu/myQuery/v1.0.0"]
                feature = ReceptorDistribution(region_name,file_urls)
                if feature:
                    self.register(feature)


if __name__ == '__main__':
    auth = Authentication.instance()
    auth.set_token('eyJhbGci....')
    extractor = ReceptorQuery()
    print(extractor)
