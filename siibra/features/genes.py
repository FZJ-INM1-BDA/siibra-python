# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from xml.etree import ElementTree
import numpy as np
import json
import siibra
from siibra import retrieval,spaces,logger
from .feature import SpatialFeature
from .extractor import FeatureExtractor
from os import path

class GeneExpression(SpatialFeature):
    """
    A spatial feature type for gene expressions.
    """

    def __init__(self,gene,space,location,expression_levels,z_scores,probe_ids,donor_info,mri_coord=None):
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
        probe_ids : list of int
            The probe_ids corresponding to each z_score element
        donor_info : dict (keys: age, race, gender, donor, speciment)
            Dictionary of donor attributes
        mri_coord : tuple  (optional)
            coordinates in original mri space
        """
        SpatialFeature.__init__(self,location,space)
        self.expression_levels = expression_levels
        self.z_scores = z_scores
        self.donor_info = donor_info
        self.gene = gene
        self.probe_ids = probe_ids
        self.mri_coord = mri_coord

    def __str__(self):
        return " ".join([
            "At ("+",".join("{:4.0f}".format(v) for v in self.location)+")",
            " ".join(["{:>7.7}:{:7.7}".format(k,str(v)) for k,v in self.donor_info.items()]),
            "Expression: ["+",".join(["%4.1f"%v for v in self.expression_levels])+"]",
            "Z-score: ["+",".join(["%4.1f"%v for v in self.z_scores])+"]"
            ])

class AllenBrainAtlasQuery(FeatureExtractor):
    """
    Interface to Allen Human Brain Atlas microarray data.
    
    This class connects to the web API of the Allen Brain Atlas:
    © 2015 Allen Institute for Brain Science. Allen Brain Atlas API. 
    Available from: brain-map.org/api/index.html
    Any use of the data needs to be in accordance with their terms of use, see
    https://alleninstitute.org/legal/terms-use/

    - We have samples from 6 different human donors. 
    - Each donor corresponds to exactly 1 specimen (tissue used for study)
    - Each sample was subject to multiple (in fact 4) different probes.
    - The probe data structures contain the list of gene expression of a
      particular gene measured in each sample. Therefore the length of the gene
      expression list in a probe coresponds to the number of samples taken in
      the corresponding donor for the given gene.
    """
    _FEATURETYPE = GeneExpression

    ALLEN_ATLAS_NOTIFICATION=\
"""For retrieving microarray data, siibra connects to the web API of
the Allen Brain Atlas (© 2015 Allen Institute for Brain Science), available
from https://brain-map.org/api/index.html. Any use of the microarray data needs
to be in accordance with their terms of use, as specified at
https://alleninstitute.org/legal/terms-use/."""
    _notification_shown=False


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
    genename_file = path.join(path.dirname(siibra.__file__),'features','gene_names.json')
    with open(genename_file,'r') as f:
        GENE_NAMES = json.load(f)

    def __init__(self,gene):
        """
        Retrieves probes IDs for the given gene, then collects the
        Microarray probes, samples and z-scores for each donor.
        TODO check that this is only called for ICBM space
        """

        FeatureExtractor.__init__(self)
        self.gene = gene

        if not self.__class__._notification_shown:
            print(self.__class__.ALLEN_ATLAS_NOTIFICATION) 
            self.__class__._notification_shown=True
        logger.info("Retrieving probe ids for gene {}".format(gene))
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
                    'age' : int(item['age']['days']/365)
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
            url,msg_if_not_cached="Retrieving specimen information for id {}".format(
                specimen_id)))
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
            donor = {k:sample['donor'][k] for k in ['name','id']}
            icbm_coord = np.matmul(
                    self._specimen[donor['name']]['donor2icbm'],
                    sample['sample']['mri']+[1] ).T

            # Create the spatial feature
            self.register( GeneExpression( 
                self.gene,
                icbm_coord, 
                spaces.MNI152_2009C_NONL_ASYM,
                expression_levels = [float(p['expression_level'][i]) for p in probes],
                z_scores = [float(p['z-score'][i]) for p in probes],
                probe_ids = [p['id'] for p in probes],
                donor_info = {**self.factors[donor['id']], **donor},
                mri_coord = sample['sample']['mri']
                ))

if __name__ == "__main__":

    featureextractor = AllenBrainAtlasQuery('GABARAPL2')
