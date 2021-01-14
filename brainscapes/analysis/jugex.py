from scipy.stats.mstats import winsorize
from statsmodels.formula.api import ols
import statsmodels.api as sm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from brainscapes.atlas import Atlas
from brainscapes import spaces,logger
from brainscapes.features import gene_names, modalities
from collections import defaultdict

class DifferentialGeneExpression:

    icbm_id = spaces.MNI_152_ICBM_2009C_NONLINEAR_ASYMMETRIC.id 

    def __init__(self,atlas: Atlas, n_rep=1000, gene_names=[], random_seed=None):
        self.n_rep = n_rep
        self.random_seed = random_seed
        self.result = None
        self.samples1 = defaultdict(dict)
        self.samples2 = defaultdict(dict)
        self.gene_names = gene_names
        spaces_supported = atlas.selected_parcellation.maps.keys()
        if self.icbm_id not in spaces_supported:
            raise Exception("Atlas provided to DifferentialGeneExpression analysis does not support the {} space in its selected parcellation {}.".format(
                self.icbm_id, atlas.selected_parcellation))
        self.atlas = atlas

    @staticmethod
    def _anova_iteration(area,zscores,donor_factors):
        """
        Run a single ANOVA iteration on the current factors.
        """
        variable_factors = {
            "area" : area,
            "zscores": zscores
        }
        mod = ols( 'zscores ~ area + specimen + age + race',
                  data=variable_factors|donor_factors ).fit()
        aov_table = sm.stats.anova_lm(mod, typ=1)
        return aov_table['F'][0]

    def run(self):

        if len(self.gene_names)==0:
            logger.warn('No candidate genes defined. Use "add_candidate_gene"')
            return

        if len(self.samples1)<1 or len(self.samples2)<1:
            logger.warn('Not enough samples found for the given genes and regions.')
            return

        # aggregate factors
        samples = self.samples1|self.samples2
        specimen = [ s['name'] for loc,s in samples.items()]
        age = [ s['age'] for loc,s in samples.items()]
        race = [ s['race'] for loc,s in samples.items()]
        area = [ s['area'] for loc,s in samples.items()]
        coords = [ loc for loc,_ in samples.items()]
        zscores = { gene_name : [s[gene_name] for loc,s in samples.items()]
                          for gene_name in self.gene_names }

        # split constant and variable factors
        donor_factors = { "specimen":specimen, "age":age, "race":race }
        brain_area = area
        averaged_zscores = zscores

        # convenience function for reuse below
        run_iteration = lambda t: self._anova_iteration(t[0],t[1],donor_factors)

        # first iteration
        Fv = np.array([ run_iteration((brain_area,mean_zscores))
                       for _,mean_zscores
                       in averaged_zscores.items()])

        # multi-threaded permutations
        if self.random_seed is not None:
            logger.info("Using random seed:",self.random_seed)
            np.random.seed(self.random_seed)
        trials = ((np.random.permutation(brain_area),mean_zscores)
                  for _,mean_zscores
                  in averaged_zscores.items()
                  for _ in range(self.n_rep-1))
        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(run_iteration, trials))
            Fm = np.array(scores).reshape((-1,self.n_rep-1)).T

        # collate result
        FWE_corrected_p = np.apply_along_axis(
                lambda arr : np.count_nonzero(arr)/self.n_rep, 0,
                Fm.max(1)[:, np.newaxis] >= np.array(Fv)
                )

        self.result = dict(zip(averaged_zscores.keys(), FWE_corrected_p))

    def add_candidate_gene(self,gene_name):
        """
        Adds a candidate gene to be used for the analysis.

        Parameters:
        -----------

        gene_name : str
            Name of a gene, as defeined in the Allen API. See
            brainscapes.features.gene_names for a full list.

        TODO on invalid parameter, we could show suggestions!
        """
        if gene_name not in gene_names:
            logger.warn("'{}' not found in the list of valid gene names.")
            return False
        self.gene_names.append(gene_name)
        return True

    def define_roi1(self,regiondef):
        """
        (Re-)Defines the first region of interest. 

        Parameters:
        -----------

        regiondef : str
            Identifier for a brain region in the selected atlas parcellation
        """
        new_samples = self._retrieve_samples(regiondef)
        if new_samples is None:
            raise Exception("Could not define ROI 2.")
        self.samples1 = new_samples

    def define_roi2(self,regiondef):
        """
        (Re-)defines the second region of interest. 

        Parameters:
        -----------

        regiondef : str
            Identifier for a brain region in the selected atlas parcellation
        """
        new_samples = self._retrieve_samples(regiondef)
        if new_samples is None:
            raise Exception("Could not define ROI 2.")
        self.samples2 = new_samples

    def _retrieve_samples(self,regiondef):
        """
        Retrieves and prepares gene expression samples for the given region
        definition.

        Parameters:
        -----------

        regiondef : str
            Identifier for a brain region in the selected atlas parcellation

        Returns: dictionary
            Gene expression data samples for the provided region
        """
        region = self.atlas.select_region(regiondef)
        if region is None:
            logger.warn("Region definition '{}' could not be matched in atlas.".format(regiondef))
            return None
        samples = defaultdict(dict)
        for gene_name in self.gene_names:
            for f in self.atlas.query_data(
                    modalities.GeneExpression,
                    gene=gene_name):
                loc = tuple(f.location)
                samples[loc] = samples[loc]|f.donor_info
                samples[loc]['area'] = region.name
                samples[loc][gene_name] =  np.mean(
                        winsorize(f.z_scores, limits=0.1))
        logger.info('{} samples found for region {}.'.format(
            len(samples), regiondef))
        return samples

