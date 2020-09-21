from xml.etree import ElementTree
import logging
import numpy as np
import json
from .. import retrieval

logging.basicConfig(level=logging.INFO)

BASE_URL = "http://api.brain-map.org/api/v2/data"
API_CALLS = {
    "probe" : BASE_URL+"/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'{product}'],gene[acronym$eq{gene}],rma::options[only$eq'probes.id']",
    "specimen" : BASE_URL+"/Specimen/query.json?criteria=[name$eq'{specimen}']&include=alignment3d",
    "microarray" : BASE_URL+"/query.json?criteria=service::human_microarray_expression[probes$in{probe_ids}][donors$eq{donor_id}]",
    "gene" : BASE_URL+"/Gene/query.json?criteria=products[abbreviation$eq'HumanMA']&num_rows=all"
    }

# there is a 1:1 mapping between donors and specimen for the 6 adult human brains
DONOR_IDS = ['15496', '14380', '15697', '9861', '12876', '10021'] 
SPECIMEN_IDS  = ['H0351.1015', 'H0351.1012', 'H0351.1016', 'H0351.2001', 'H0351.1009', 'H0351.2002']

# load gene names
genes = json.loads(retrieval.cached_get(API_CALLS['gene'],
        "Gene acronyms not found in cache. Retrieving list of all gene acronyms from Allen Atlas now. This may take a minute."))
GENE_NAMES = {g['acronym']:g['name'] for g in genes['msg']}

class AllenGeneExpressions:
    """
    Interface to Allen Human Brain Atlas Gene Expressions
    TODO add Allen copyright clause
    """

    def __init__(self):
        pass

    @staticmethod
    def _retrieve_specimen(specimen):
        """
        Retrieves information about each human specimen. Called at object construction.
        """
        url = API_CALLS['specimen'].format(specimen=specimen)
        response = json.loads(retrieval.cached_get(
            url,msg_if_not_cached="Retrieving specimen information for id {}".format(specimen)))
        if not response['success']:
            raise Exception('Invalid response when retrieving specimen information: {}'.format( url))
        # we ask for 1 specimen, so list should have length 1
        assert(len(response['msg'])==1)
        return response['msg'][0]

    def filter_by_coordinate(self,mask,space):
        """
        Filters the feature list by their coordinate in a given space. Only
        those features are kept which are located at nonzero positions of the
        mask.
        """

    def retrieve_gene(self,gene,product='HumanMA'): 
        """
        Retrieves all probes IDs for the given gene, then collects the
        Microarray probes, samples and z-scores for each donor.
        TODO check that this is only called for ICBM space
        """
        logging.info("Retrieving probe ids for gene {}".format(gene))
        url = API_CALLS['probe'].format(gene=gene,product=product)
        response = retrieval.cached_get(url) 
        root = ElementTree.fromstring(response)
        num_probes = int(root.attrib['total_rows'])
        probe_ids = [int(root[0][i][0].text) for i in range(num_probes)]
        for donor_id in DONOR_IDS:
            probes,samples,zscores = AllenGeneExpressions._retrieve_microarray(donor_id,probe_ids)
            print("{:>15} {} probes {} samples {}x{} zscores ".format(
                "donor "+donor_id, len(probes), len(samples), zscores.shape[0],zscores.shape[1] ))

    @staticmethod
    def _retrieve_microarray(donor_id, probe_ids):
        """
        Retrieve microarray data for a given donor and probe, and compute the
        MRI position of the corresponding tissue block.
        """
        url = API_CALLS['microarray'].format(
                probe_ids=','.join([str(id) for id in probe_ids]),
                donor_id=donor_id)
        response = json.loads(retrieval.cached_get(url))
        if not response['success']:
            raise Exception('Invalid response when retrieving microarray data: {}'.format( url))
        probes,samples = [response['msg'][k] for  k in ['probes','samples']] 
        zscores = np.array([ probe['z-score'] for probe in probes],dtype=np.float64).T

        # convert MRI coordinates of the samples to ICBM MNI152 space
        specimen = {}
        for probe,sample in zip(probes,samples):

            spcm_id=sample['donor']['name']

            # cached retrieval of specimen information - there are only 6 of them
            if spcm_id not in specimen.keys():
                specimen[spcm_id] = AllenGeneExpressions._retrieve_specimen(spcm_id)
                T = specimen[spcm_id]['alignment3d']
                specimen[spcm_id]['donor2icbm'] = np.array([
                    [T['tvr_00'], T['tvr_01'], T['tvr_02'], T['tvr_09']],
                    [T['tvr_03'], T['tvr_04'], T['tvr_05'], T['tvr_10']],
                    [T['tvr_06'], T['tvr_07'], T['tvr_08'], T['tvr_11']] ])

            donor2icbm = specimen[spcm_id]['donor2icbm']
            donor_coord = sample['sample']['mri']+[1]
            icbm_coord = np.matmul(donor2icbm, donor_coord).T
            sample['icbm_coord'] = icbm_coord

        return probes,samples,zscores

if __name__ == "__main__":

    genes = AllenGeneExpressions()
    genes.retrieve_gene('GABARAPL2')
