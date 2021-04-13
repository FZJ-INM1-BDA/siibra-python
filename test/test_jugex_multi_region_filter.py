import os
import unittest
import siibra as sb
from siibra.analysis import DifferentialGeneExpression
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

class TestJugexMultiRegionFilter(unittest.TestCase):

    def test_region_filter(self):
        sb.logger.setLevel("INFO")
        atlas = sb.atlases.MULTILEVEL_HUMAN_ATLAS
        atlas.select_parcellation(sb.parcellations.JULICH_BRAIN_PROBABILISTIC_CYTOARCHITECTONIC_MAPS_V2_5)
        atlas.enable_continuous_map_thresholding(0.2)
        jugex = DifferentialGeneExpression(atlas)

        input_dict = {
            (1.1111, 2.222, 3.333, "region-name"): {},
            (1.2345, 2.345, 3.000, "region-name"): {},
            (1.1111, 2.222, 3.333, "another region-name"): {}
        }

        expected_dict = {
            (1.1111, 2.222, 3.333, "region-name"): {},
            (1.2345, 2.345, 3.000, "region-name"): {}
        }

        self.assertEqual(jugex._filter_samples(input_dict), expected_dict)


if __name__ == "__main__":
    unittest.main()
