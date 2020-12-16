import unittest
from brainscapes.atlas import REGISTRY


class TestAtlas(unittest.TestCase):

    def test_parcellations(self):
        atlas = REGISTRY.MULTILEVEL_HUMAN_ATLAS
        parcellations = atlas.parcellations
        print(len(parcellations))
        print(parcellations)
        self.assertTrue(len(parcellations) == 10)


if __name__ == "__main__":
    unittest.main()
