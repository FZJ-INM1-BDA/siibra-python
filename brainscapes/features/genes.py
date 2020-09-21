from xml.etree import ElementTree
import logging
import numpy as np
import json
from brainscapes import retrieval
from brainscapes.definitions import spaces
from brainscapes.features.feature import SpatialFeature,FeaturePool

logging.basicConfig(level=logging.INFO)

class GeneExpressionFeature(SpatialFeature):
    """
    A spatial feature type for gene expressions.
    """

    def __init__(self,space,location,expression_levels,z_scores,factors):
        """
        Construct the spatial feature for gene expressions measured in a sample.

        Parameters:
        -----------
        space : str       
            Name of 3D reference template space in which this feature is defined
        location : tuple of float    
            3D location of the gene expression sample in physical coordinates
            of the given space
        expression_levels : list of float
            expression levels measured in possibly multiple probes of the same sample
        z_scores : list of float
            z scores measured in possibly multiple probes of the same sample
        factors : dict (keys: age, race, gender)
            Dictionary of social factors of the donor 
        """
        SpatialFeature.__init__(self,location,space)
        self.expression_levels = expression_levels
        self.z_scores = z_scores
        self.factors = factors

class AllenBrainAtlasQuery(FeaturePool):
    """
    Interface to Allen Human Brain Atlas Gene Expressions
    TODO add Allen copyright clause

    To better understand the principles:
    - We have samples from 6 different human donors. 
    - Each donor corresponds to exactly 1 specimen (tissue used for study)
    - Each sample was subject to multiple (in fact 4) different probes.
    - The probe data structures contain the list of gene expression of a
      particular gene measured in each sample. Therefore the length of the gene
      expression list in a probe coresponds to the number of samples taken in
      the corresponding donor for the given gene.
    """

    _BASE_URL = "http://api.brain-map.org/api/v2/data"
    _QUERY = {
        "probe" : _BASE_URL+"/query.xml?criteria=model::Probe,rma::criteria,[probe_type$eq'DNA'],products[abbreviation$eq'HumanMA'],gene[acronym$eq{gene}],rma::options[only$eq'probes.id']",
        "specimen" : _BASE_URL+"/Specimen/query.json?criteria=[name$eq'{specimen_id}']&include=alignment3d",
        "microarray" : _BASE_URL+"/query.json?criteria=service::human_microarray_expression[probes$in{probe_ids}][donors$eq{donor_id}]",
        "gene" : _BASE_URL+"/Gene/query.json?criteria=products[abbreviation$eq'HumanMA']&num_rows=all",
        "factors" : _BASE_URL+"/query.json?criteria=model::Donor,rma::criteria,products[id$eq2],rma::include,age,rma::options[only$eq%27donors.id,dono  rs.name,donors.race_only,donors.sex%27]"
        }

    # there is a 1:1 mapping between donors and specimen for the 6 adult human brains
    _DONOR_IDS = [
            '15496', 
            '14380', 
            '15697', 
            '9861', 
            '12876', 
            '10021'] 
    _SPECIMEN_IDS  = [
            'H0351.1015', 
            'H0351.1012', 
            'H0351.1016', 
            'H0351.2001', 
            'H0351.1009', 
            'H0351.2002']

    # load gene names
    genes = json.loads(retrieval.cached_get(_QUERY['gene'],
            "Gene acronyms not found in cache. Retrieving list of all gene acronyms from Allen Atlas now. This may take a minute."))
    GENE_NAMES = {g['acronym']:g['name'] for g in genes['msg']}

    def __init__(self,gene):
        """
        Retrieves probes IDs for the given gene, then collects the
        Microarray probes, samples and z-scores for each donor.
        TODO check that this is only called for ICBM space
        """

        FeaturePool.__init__(self)

        # get probe ids for the given gene
        logging.info("Retrieving probe ids for gene {}".format(gene))
        url = self._QUERY['probe'].format(gene=gene)
        response = retrieval.cached_get(url) 
        root = ElementTree.fromstring(response)
        num_probes = int(root.attrib['total_rows'])
        probe_ids = [int(root[0][i][0].text) for i in range(num_probes)]

        # get specimen information
        self._specimen = {
                spcid:self._retrieve_specimen(spcid) 
                for spcid in self._SPECIMEN_IDS}
        response = json.loads(retrieval.cached_get(self._QUERY['factors']))
        self.factors = {
                item['id']: {
                    'race' : item['race_only'],
                    'gender' : item['sex'],
                    'age' : item['age']['days']/365
                    }
                for item in response['msg'] }

        # get expression levels and z_scores for the gene
        for donor_id in self._DONOR_IDS:
            self._retrieve_microarray(donor_id,probe_ids)

    def _retrieve_specimen(self,specimen_id):
        """
        Retrieves information about a human specimen. 
        """
        url = self._QUERY['specimen'].format(specimen_id=specimen_id)
        response = json.loads(retrieval.cached_get(
            url,msg_if_not_cached="Retrieving specimen information for id {}".format(specimen_id)))
        if not response['success']:
            raise Exception('Invalid response when retrieving specimen information: {}'.format( url))
        # we ask for 1 specimen, so list should have length 1
        assert(len(response['msg'])==1)
        specimen = response['msg'][0]
        T = specimen['alignment3d']
        specimen['donor2icbm'] = np.array([
            [T['tvr_00'], T['tvr_01'], T['tvr_02'], T['tvr_09']],
            [T['tvr_03'], T['tvr_04'], T['tvr_05'], T['tvr_10']],
            [T['tvr_06'], T['tvr_07'], T['tvr_08'], T['tvr_11']] ])
        return specimen

    def _retrieve_microarray(self,donor_id, probe_ids):
        """
        Retrieve microarray data for several probes of a given donor, and
        compute the MRI position of the corresponding tissue block in the ICBM
        152 space to generate a SpatialFeature object for each sample.
        """

        # query the microarray data for this donor
        url = self._QUERY['microarray'].format(
                probe_ids=','.join([str(id) for id in probe_ids]),
                donor_id=donor_id)
        response = json.loads(retrieval.cached_get(url))
        if not response['success']:
            raise Exception('Invalid response when retrieving microarray data: {}'.format( url))

        # store probes
        probes,samples = [response['msg'][n] for n in ['probes','samples']]

        # store samples. Convert their MRI coordinates of the samples to ICBM
        # MNI152 space
        for i,sample in enumerate(samples):

            # coordinate conversion to ICBM152 standard space
            spcid,donor_id = [sample['donor'][k] for k in ['name','id']]
            icbm_coord = np.matmul(
                    self._specimen[spcid]['donor2icbm'],
                    sample['sample']['mri']+[1] ).T

            # Create the spatial feature
            self.features.append( 
                    GeneExpressionFeature( 
                        icbm_coord, 
                        'MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC',
                        expression_levels = [float(p['expression_level'][i]) 
                            for p in probes],
                        z_scores = [float(p['z-score'][i]) for p in probes],
                        factors = self.factors[donor_id]
                        )
                    )

if __name__ == "__main__":

    featurepool = AllenBrainAtlasQuery('GABARAPL2')
