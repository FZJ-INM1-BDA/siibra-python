import os
import unittest
from siibra import atlases
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]


class TestSelectionBrainRegions(unittest.TestCase):

    def test_select_brain_regions(self):
        atlas = atlases.MULTILEVEL_HUMAN_ATLAS
        region = atlas.get_region(parcellation='2.9', region='v1')
        jubrain = region.parcellation
        self.assertEqual(region.name, 'Area hOc1 (V1, 17, CalcS)')
        self.assertTrue(len(region.children) == 2)

        print('v1 includes the left and right hemisphere!')
        print(repr(region))

        # we can be more specific easily
        region = atlas.get_region('v1 left')
        print("Selected region from 'v1 left' is", region.name)
        self.assertEqual(region.name, 'Area hOc1 (V1, 17, CalcS) left')
        self.assertTrue(len(region.children) == 0)

        # we can also auto-complete on the 'regionnames' attribute of the atlas
        # - this immediately leads to a unique selection
        jubrain = region.parcellation
        region = atlas.get_region(jubrain.names.AREA_HOC1_V1_17_CALCS_LEFT)
        self.assertEqual(region.name, 'Area hOc1 (V1, 17, CalcS) left')
        self.assertTrue(len(region.children) == 0)


if __name__ == "__main__":
    unittest.main()
