import unittest
from brainscapes.atlas import REGISTRY
from brainscapes.features import modalities
import brainscapes as bs
from brainscapes import retrieval


class TestExtractTransmitterReceptorDensities(unittest.TestCase):

    def test_extract_densities(self):
        atlas = REGISTRY.MULTILEVEL_HUMAN_ATLAS
        atlas.select_parcellation(bs.parcellations.JULICH_BRAIN_PROBABILISTIC_CYTOARCHITECTONIC_MAPS_V2_5)
        atlas.select_region(atlas.regionnames.AREA_HOC1_V1_17_CALCS_LEFT_HEMISPHERE)
        features = atlas.query_data(modalities.ReceptorDistribution)
        self.assertTrue(len(features) == 1)
        self.assertEqual(features[0].name, 'Density measurements of different receptors for Area hOc1 (V1, 17, CalcS) [human, v1.0]')

        query = bs.features.receptors.ReceptorQuery()
        self.assertTrue(len(query.features) == 69)


if __name__ == "__main__":
    unittest.main()
