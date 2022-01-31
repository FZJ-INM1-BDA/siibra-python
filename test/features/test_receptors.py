import unittest
import pytest

import siibra
from siibra.features.receptors import RECEPTOR_SYMBOLS, unify_stringlist, ReceptorQuery, ReceptorDistribution
import tempfile


class TestReceptors(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    def test_receptor_symbols(self):
        self.assertEqual(len(RECEPTOR_SYMBOLS), 16)

    def test_unify_string_list(self):
        unified_list = unify_stringlist(['a', 'b', 'a', 'c', 'b', 'a'])
        self.assertEqual(unified_list, ['a', 'b', 'a*', 'c', 'b*', 'a**'])

    def test_unify_error_if_not_all_string_items(self):
        with self.assertRaises(AssertionError):
            unify_stringlist(['a', 'b', 1])
        self.assertTrue('Assertion error is expected')

receptor_query = ReceptorQuery()

def test_get_hoc1_left_density():
    atlas = siibra.atlases['human']
    region = atlas.get_region("hoc1 left", parcellation="2.9")
    features = siibra.get_features(region, siibra.modalities.ReceptorDistribution)
    assert len(features) == 1, f"expect only 1 result from getting hoc1 left receptor, but got {len(features)}"
    expected_name = "Density measurements of different receptors for Area hOc1 (V1, 17, CalcS) [human, v1.0]"
    assert features[0].name == expected_name, f"name of fetched receptor does not match. Expected {expected_name}, got {features[0].name}"

regions_has_receptor = ["hOc1", "hOc2", "44"]

def test_no_receptor_data():
    assert len(receptor_query.features) >= 42, f"expect at least 42 receptor query data, but got {len(receptor_query.features)}"

@pytest.mark.parametrize("region_spec", regions_has_receptor)
def test_get_region_receptor(region_spec:str):
    atlas = siibra.atlases['human']
    r = atlas.get_region(region_spec)
    matched_features = [
        f for f in receptor_query.features if r.matches(f.regionspec)
    ]
    assert len(matched_features) > 0, f"expect at least one receptor query matching {region_spec}, but had none."


@pytest.mark.parametrize('receptor', receptor_query.features)
def test_receptor_to_model(receptor: ReceptorDistribution):
    receptor.to_model()

if __name__ == "__main__":
    unittest.main()
