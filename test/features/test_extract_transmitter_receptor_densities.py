import os
import unittest
import siibra
from test.get_token import get_token

token = get_token()
os.environ["HBP_AUTH_TOKEN"] = token["access_token"]


class TestExtractTransmitterReceptorDensities(unittest.TestCase):
    def test_extract_densities(self):
        atlas = siibra.atlases["human"]
        region = atlas.get_region("hoc1 left", parcellation="2.9")
        features = siibra.get_features(region, siibra.modalities.ReceptorDistribution)
        self.assertTrue(len(features) == 1)
        self.assertEqual(
            features[0].name,
            "Density measurements of different receptors for Area hOc1 (V1, 17, CalcS) [human, v1.0]",
        )

        regions = ["hOc1", "hOc2", "44"]
        query = siibra.features.receptors.ReceptorQuery()
        self.assertTrue(len(query.features) >= 42)
        for q in regions:
            r = atlas.get_region(q)
            matched_features = [
                f for f in query.features if r.matches(f.regionspec)
            ]
            self.assertGreater(len(matched_features), 0)


if __name__ == "__main__":
    unittest.main()
