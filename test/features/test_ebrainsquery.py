from typing import List
import unittest
import siibra
import pytest
from siibra.core import Parcellation, Atlas, Region
from siibra.features.feature import Feature

class TestEbrainsQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atlas = siibra.atlases.MULTILEVEL_HUMAN_ATLAS
        region = atlas.get_region("hoc1 left")
        cls.feat = siibra.get_features(region, siibra.modalities.EbrainsRegionalDataset)

    def test_some_result_returned(self):
        assert len(self.feat) > 0

    def test_no_duplicates_returned(self):
        ids = [f.id for f in self.feat]
        assert len(self.feat) == len(list(set(ids)))


parameter = [
    ('rat', 'v3', 'neocortex', {
        'exclude': ['DiFuMo atlas (512 dimensions)'],
        'include': ['3D high resolution SRXTM image data of cortical vasculature of rat brain.']
    })
]

@pytest.mark.parametrize('atlas_id,parc_id,region_id,inc_exc', parameter)
def test_species(atlas_id,parc_id,region_id,inc_exc):
    atlas:Atlas = siibra.atlases[atlas_id]
    parc:Parcellation = atlas.parcellations[parc_id]
    r:Region = parc.decode_region(region_id)
    features: List[Feature] = siibra.get_features(r, 'ebrains')
    feature_names = [f.name for f in features]

    excludes: List[str] = inc_exc.get('exclude')
    includes: List[str] = inc_exc.get('include')
    assert all(exc not in feature_names for exc in excludes)
    assert all(inc in feature_names for inc in includes)

if __name__ == "__main__":
    unittest.main()
