import unittest
import siibra


class TestEbrainsQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atlas = siibra.atlases["human"]
        region = atlas.get_region("hoc1 left")
        cls.feat = siibra.get_features(region, siibra.modalities.EbrainsRegionalDataset)

    def test_some_result_returned(self):
        assert len(self.feat) > 0

    def test_no_duplicates_returned(self):
        ids = [f.id for f in self.feat]
        assert len(self.feat) == len(list(set(ids)))


if __name__ == "__main__":
    unittest.main()
