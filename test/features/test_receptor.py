import unittest
from siibra import atlases


class TestAtlas(unittest.TestCase):

    def test_parcellations(self):
        atlas = atlases.MULTILEVEL_HUMAN_ATLAS
        parcellations = atlas.parcellations
        self.assertTrue(len(parcellations) >= 11)


if __name__ == "__main__":
    unittest.main()
