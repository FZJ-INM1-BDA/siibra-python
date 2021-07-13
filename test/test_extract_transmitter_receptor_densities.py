import os
import unittest
from siibra.atlas import REGISTRY
from siibra.features import modalities
import siibra as sb
from siibra import retrieval
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

class TestExtractTransmitterReceptorDensities(unittest.TestCase):

    def test_extract_densities(self):
        atlas = sb.atlases['human']
        atlas.select_parcellation('2.9')
        atlas.select_region('hoc1 left')
        features = atlas.get_features(modalities.ReceptorDistribution)
        self.assertTrue(len(features) == 1)
        self.assertEqual(features[0].name, 'Density measurements of different receptors for Area hOc1 (V1, 17, CalcS) [human, v1.0]')

        regions = ['hOc1', 'hOc2', 'IFG']
        query = sb.features.receptors.ReceptorQuery(atlas)
        self.assertTrue(len(query.features) >= 42)
        for q in regions:
            matched_features = [f for f in query.features if f.region.matches(q)]
            self.assertGreater(len(matched_features),0)


if __name__ == "__main__":
    unittest.main()
