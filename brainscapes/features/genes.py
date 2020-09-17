
import requests
from xml.etree import ElementTree
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

class AllenGenes:
    """
    Interface to Allen Human Brain Atlas Gene Expressions
    TODO add Allen copyright clause
    """

    BASE_URL = "http://api.brain-map.org/api/v2/data"
    PROBE_URL      = BASE_URL+"/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$eq{gene}],rma::options[only$eq'probes.id']" 
    SPECIMEN_URL   = BASE_URL+"/Specimen/query.json?criteria=[name$eq'{specimen}']&include=alignment3d"
    MICROARRAY_URL = BASE_URL+"/query.json?criteria=service::human_microarray_expression[probes$in{probe_ids}][donors$eq{donor_id}]"
    DONOR_IDS = ['15496', '14380', '15697', '9861', '12876', '10021'] 
    SPECIMEN  = ['H0351.1015', 'H0351.1012', 'H0351.1016', 'H0351.2001', 'H0351.1009', 'H0351.2002']

    _specimen = []
    
    def __init__(self):
        for specimen in AllenGenes.SPECIMEN:
            self._specimen.append(
                    self.__retrieve_specimen(specimen) )

    def retrieve_gene(self, gene): 
        logging.info("Retrieving probe ids for gene {}".format(gene))
        url = self.PROBE_URL.format(gene=gene)
        response = requests.get(url) 
        root = ElementTree.fromstring(response.text)
        num_probes = int(root.attrib['total_rows'])
        probe_ids = [int(root[0][i][0].text) for i in range(num_probes)]
        for donor_id in AllenGenes.DONOR_IDS:
            probes,samples,zscores = self.__retrieve_microarray(donor_id,probe_ids)
            print("{:>15} {} probes {} samples {}x{} zscores ".format(
                "donor "+donor_id,
                len(probes), len(samples),
                zscores.shape[0],zscores.shape[1] ))

    def __retrieve_specimen(self, specimen):
        logging.info("Retrieving specimen information for id {}".format(specimen))
        url = self.SPECIMEN_URL.format(specimen=specimen)
        response = requests.get(url).json()
        if not response['success']:
            raise Exception('Invalid response when retrieving specimen information: {}'.format( url))
        # we ask for 1 specimen, so list should have length 1
        assert(len(response['msg'])==1)
        return response['msg'][0]

    def __retrieve_microarray(self,donor_id, probe_ids):
        url = self.MICROARRAY_URL.format(
                probe_ids=','.join([str(id) for id in probe_ids]),
                donor_id=donor_id)
        response = requests.get(url).json()
        if not response['success']:
            raise Exception('Invalid response when retrieving microarray data: {}'.format( url))
        probes,samples = [response['msg'][k] for  k in ['probes','samples']] 
        zscores = np.array([ probe['z-score'] for probe in probes],dtype=np.float64).T
        return probes,samples,zscores

if __name__ == "__main__":

    genes = AllenGenes()
    genes.retrieve_gene('GABARAPL2')

