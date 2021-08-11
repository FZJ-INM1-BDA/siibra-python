import os
import unittest
from siibra import atlases
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

class TestSelectionBrainRegions(unittest.TestCase):

    def test_select_brain_regions(self):
        atlas = atlases.MULTILEVEL_HUMAN_ATLAS
        atlas.select(parcellation='2.9',region='v1')
        jubrain=atlas.selection.parcellation
        self.assertEqual(atlas.selection.region.name, 'Area hOc1 (V1, 17, CalcS)')
        self.assertTrue(len(atlas.selection.region.children) == 2)

        print('v1 includes the left and right hemisphere!')
        print(repr(atlas.selection.region))

        # we can be more specific easily
        atlas.select(region='v1 left')
        print("Selected region from 'v1 left' is", atlas.selection.region.name)
        self.assertEqual(atlas.selection.region.name, 'Area hOc1 (V1, 17, CalcS) left')
        self.assertTrue(len(atlas.selection.region.children) == 0)

        # we can also auto-complete on the 'regionnames' attribute of the atlas
        # - this immediately leads to a unique selection
        jubrain=atlas.selection.parcellation
        atlas.select(region=jubrain.names.AREA_HOC1_V1_17_CALCS_LEFT)
        self.assertEqual(atlas.selection.region.name, 'Area hOc1 (V1, 17, CalcS) left')
        self.assertTrue(len(atlas.selection.region.children) == 0)


if __name__ == "__main__":
    unittest.main()
