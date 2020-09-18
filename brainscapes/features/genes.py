
import requests
from xml.etree import ElementTree
import logging
import numpy as np
import json
from tempfile import mkdtemp
import hashlib
from os import path,makedirs
#from .feature import RegionFeature
logging.basicConfig(level=logging.INFO)

class AllenGeneExpressions:
    """
    Interface to Allen Human Brain Atlas Gene Expressions
    TODO add Allen copyright clause
    """

    BASE_URL = "http://api.brain-map.org/api/v2/data"
    PROBE_URL      = BASE_URL+"/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$eq{gene}],rma::options[only$eq'probes.id']" 
    SPECIMEN_URL   = BASE_URL+"/Specimen/query.json?criteria=[name$eq'{specimen}']&include=alignment3d"
    MICROARRAY_URL = BASE_URL+"/query.json?criteria=service::human_microarray_expression[probes$in{probe_ids}][donors$eq{donor_id}]"
    # there is a 1:1 mapping between donors and specimen for the 6 adult human brains
    DONOR_IDS = ['15496', '14380', '15697', '9861', '12876', '10021'] 
    SPECIMEN_IDS  = ['H0351.1015', 'H0351.1012', 'H0351.1016', 'H0351.2001', 'H0351.1009', 'H0351.2002']

    def __init__(self,cache=None):
        # Set cache dir for downloads
        self.__cache_dir = mkdtemp() if cache is None else cache
        if not path.isdir(self.__cache_dir):
            makedirs(self.__cache_dir)
        logging.info("Cachedir: {}".format(self.__cache_dir))


    def __cached_get(self,url):
        """
        Performs a requests.get if the result is not yet available in the local
        cache, otherwise returns the result from the cache.
        TODO we might extend this as a general tool for the brainscapes library, and make it a decorator
        """
        url_hash = hashlib.sha256(url.encode('ascii')).hexdigest()
        hash_target = path.join(self.__cache_dir,url_hash)

        if path.isfile(hash_target):
            # This URL target is already in the cache - just return it
            logging.debug("Returning cached response of url {} at {}".format(url,hash_target))
            with open(hash_target,'r') as f:
                response = f.read()
        else:
            logging.debug("Downloading {}".format(url))
            response = requests.get(url).text
            with open(hash_target,'w') as f:
                logging.debug("Caching response to {}".format(hash_target))
                f.write(response)
            with open(hash_target+".url",'w') as f:
                f.write(url)
        return response

    def __retrieve_specimen(self, specimen):
        logging.info("Retrieving specimen information for id {}".format(specimen))
        url = self.SPECIMEN_URL.format(specimen=specimen)
        response = json.loads(self.__cached_get(url))
        if not response['success']:
            raise Exception('Invalid response when retrieving specimen information: {}'.format( url))
        # we ask for 1 specimen, so list should have length 1
        assert(len(response['msg'])==1)
        return response['msg'][0]

    def retrieve_gene(self, gene): 
        """
        Retrieves all probes IDs for the given gene, then collects the
        Microarray probes, samples and z-scores for each donor.
        """
        logging.info("Retrieving probe ids for gene {}".format(gene))
        url = self.PROBE_URL.format(gene=gene)
        response = self.__cached_get(url) 
        root = ElementTree.fromstring(response)
        num_probes = int(root.attrib['total_rows'])
        probe_ids = [int(root[0][i][0].text) for i in range(num_probes)]
        for donor_id in AllenGeneExpressions.DONOR_IDS:
            probes,samples,zscores = self.__retrieve_microarray(donor_id,probe_ids)
            print("{:>15} {} probes {} samples {}x{} zscores ".format(
                "donor "+donor_id,
                len(probes), len(samples),
                zscores.shape[0],zscores.shape[1] ))

    def __retrieve_microarray(self,donor_id, probe_ids):
        """
        Retrieve microarray data for a given donor and probe, and compute the
        MRI position of the corresponding tissue block.
        """
        url = self.MICROARRAY_URL.format(
                probe_ids=','.join([str(id) for id in probe_ids]),
                donor_id=donor_id)
        response = json.loads(self.__cached_get(url))
        if not response['success']:
            raise Exception('Invalid response when retrieving microarray data: {}'.format( url))
        probes,samples = [response['msg'][k] for  k in ['probes','samples']] 
        zscores = np.array([ probe['z-score'] for probe in probes],dtype=np.float64).T

        # convert MRI coordinates of the samples to ICBM MNI152 space
        specimen = {}
        for probe,sample in zip(probes,samples):
            spcm_id=sample['donor']['name']
            if spcm_id not in specimen.keys():
                specimen[spcm_id] = self.__retrieve_specimen(spcm_id)
            T = specimen[spcm_id]['alignment3d']
            pass

        return probes,samples,zscores

if __name__ == "__main__":

    genes = AllenGeneExpressions('/tmp/gene.py')
    genes.retrieve_gene('GABARAPL2')
