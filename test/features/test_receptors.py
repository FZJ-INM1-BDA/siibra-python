import unittest
import pytest

import siibra

import tempfile


class TestReceptors(unittest.TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode="w", delete=False)

    def test_receptor_symbols(self):
        self.assertEqual(len(siibra.vocabularies.RECEPTOR_SYMBOLS), 16)

    def test_unify_string_list(self):
        unified_list = siibra.commons.unify_stringlist(["a", "b", "a", "c", "b", "a"])
        self.assertEqual(unified_list, ["a", "b", "a*", "c", "b*", "a**"])

    def test_unify_error_if_not_all_string_items(self):
        with self.assertRaises(AssertionError):
            siibra.commons.unify_stringlist(["a", "b", 1])
        self.assertTrue("Assertion error is expected")


def test_get_hoc1_left_density():
    region = siibra.get_region("julich 2.9", "hoc1 left")
    features = siibra.features.get(
        region, siibra.features.molecular.ReceptorDensityFingerprint
    )
    assert (
        len(features) == 1
    ), "expect only 1 result from getting hoc1 left receptor, but got {len(features)}"
    expected_substrings = ["receptor density"]
    assert all(
        s in features[0].name.lower() for s in expected_substrings
    ), f"name of fetched receptor unexpected. '{features[0].name}' was expected to contain {', '.join(expected_substrings)}"


regions_has_receptor = ["hOc1", "hOc2", "44"]
fingerprints = siibra.features.molecular.ReceptorDensityFingerprint._get_instances()


def test_no_receptor_data():
    human_receptor_fingerprints = [
        f
        for f in fingerprints
        if siibra.commons.Species.HOMO_SAPIENS in f.anchor.species
    ]
    assert (
        len(human_receptor_fingerprints) >= 42
    ), f"expect at least 42 receptor query data, but found only {len(human_receptor_fingerprints)}"


@pytest.mark.parametrize("region_spec", regions_has_receptor)
def test_get_region_receptor(region_spec: str):
    atlas = siibra.atlases["human"]
    r = atlas.get_region(region_spec)
    matched_features = [f for f in fingerprints if f.anchor.matches(r)]
    assert (
        len(matched_features) > 0
    ), f"expect at least one receptor query matching {region_spec}, but had none."


profiles = siibra.features.molecular.ReceptorDensityProfile._get_instances()


def test_receptor_density_profile_shape():
    for f in profiles:
        assert len(f.data.columns) == 1 and len(f.data.index) == 101


if __name__ == "__main__":
    unittest.main()
