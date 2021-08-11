import unittest
from siibra import atlases, modalities

class TestEbrainsQuery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        atlas = atlases["human"]
        atlas.select(region='hoc1 left')
        cls.feat=atlas.selection.get_features(modalities.EbrainsRegionalDataset)

    def test_some_result_returned(self):
        assert len(self.feat) > 0

    def test_no_duplicates_returned(self):
        ids=[f.id for f in self.feat]
        assert len(self.feat) == len(list(set(ids)))

if __name__ == "__main__":
    unittest.main()
