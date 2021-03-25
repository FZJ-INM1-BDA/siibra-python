import os
import unittest
from brainscapes.atlas import REGISTRY
from brainscapes.features import modalities
import brainscapes as bs
from brainscapes import retrieval
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

class TestExtractTransmitterReceptorDensities(unittest.TestCase):

    def test_extract_densities(self):
        atlas = REGISTRY.MULTILEVEL_HUMAN_ATLAS
        atlas.select_parcellation(bs.parcellations.JULICH_BRAIN_PROBABILISTIC_CYTOARCHITECTONIC_MAPS_V2_5)
        atlas.select_region(atlas.regionnames.AREA_HOC1_V1_17_CALCS_LEFT_HEMISPHERE)
        features = atlas.query_data(modalities.ReceptorDistribution)
        self.assertTrue(len(features) == 1)
        self.assertEqual(features[0].name, 'Density measurements of different receptors for Area hOc1 (V1, 17, CalcS) [human, v1.0]')

        regions = ['hOc1', 'hOc2', 'IFG']
        query = bs.features.receptors.ReceptorQuery()
        self.assertTrue(len(query.features) >= 69)
        for q in regions:
            matched_features = [r.region for r in query.features if q in r.region]
            self.assertGreater(len(matched_features),0)


if __name__ == "__main__":
    unittest.main()
