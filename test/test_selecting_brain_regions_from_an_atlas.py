import os
import unittest
from siibra.atlas import REGISTRY
import siibra as sb
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

class TestSelectionBrainRegions(unittest.TestCase):

    def test_select_brain_regions(self):
        atlas = REGISTRY.MULTILEVEL_HUMAN_ATLAS
        atlas.select_parcellation(sb.parcellations.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_2_5)
        # we can just give a string and see if the system can disambiguiate it
        atlas.select_region('v1')
        print("Selected region from 'v1' is", atlas.selected_region)
        self.assertEqual(str(atlas.selected_region), 'Area hOc1 (V1, 17, CalcS)')
        self.assertTrue(len(atlas.selected_region.children) == 2)

        print('v1 includes the left and right hemisphere!')
        print(repr(atlas.selected_region))

        # we can be more specific easily
        atlas.select_region('v1 left')
        print("Selected region from 'v1 left' is", atlas.selected_region)
        self.assertEqual(str(atlas.selected_region), 'Area hOc1 (V1, 17, CalcS) - left hemisphere')
        self.assertTrue(len(atlas.selected_region.children) == 0)

        # we can also auto-complete on the 'regionnames' attribute of the atlas
        # - this immediately leads to a unique selection
        atlas.select_region(atlas.regionnames.AREA_HOC1_V1_17_CALCS_LEFT_HEMISPHERE)
        self.assertEqual(str(atlas.selected_region), 'Area hOc1 (V1, 17, CalcS) - left hemisphere')
        self.assertTrue(len(atlas.selected_region.children) == 0)


if __name__ == "__main__":
    unittest.main()
