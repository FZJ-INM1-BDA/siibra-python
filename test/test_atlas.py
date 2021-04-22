import os
import unittest
from siibra.atlas import REGISTRY
from siibra import atlas
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]

class TestAtlas(unittest.TestCase):

    JSON_ATLAS_ID = "json_1337"
    JSON_ATLAS_NAME = "JSON Human Atlas"
    atlas_as_json = {
        "@id": JSON_ATLAS_ID,
        "name": JSON_ATLAS_NAME,
        "order": 1,
        "spaces": [
            "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2",
        ],
        "parcellations": [
            "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-25",
            "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579",
        ]
    }

    @classmethod
    def setUpClass(cls):
        cls.atlas = REGISTRY.MULTILEVEL_HUMAN_ATLAS
        cls.ATLAS_NAME = 'Multilevel Human Atlas'

    def test_atlas_init(self):
        a = atlas.Atlas('juelich/iav/atlas/v1.0.0/1', self.ATLAS_NAME)
        self.assertEqual(a.name, self.ATLAS_NAME)
        self.assertEqual(a.key, 'MULTILEVEL_HUMAN_ATLAS')
        self.assertEqual(a.id, 'juelich/iav/atlas/v1.0.0/1')
        self.assertIsNone(a.selected_region)
        self.assertIsNone(a.selected_parcellation)

    def test_parcellations(self):
        parcellations = self.atlas.parcellations
        self.assertTrue(len(parcellations) >= 11)
        self.atlas.select_parcellation(self.atlas.parcellations[0])
        self.assertIsNotNone(self.atlas.selected_parcellation)

    def test_spaces(self):
        spaces = self.atlas.spaces
        self.assertTrue(len(spaces) == 3)

    def test_to_string(self):
        atlas_str = str(self.atlas)
        self.assertEqual(atlas_str, self.ATLAS_NAME)

    def test_from_json(self):
        json_atlas = atlas.Atlas.from_json(self.atlas_as_json)
        self.assertEqual(json_atlas['name'], self.JSON_ATLAS_NAME)
        self.assertEqual(json_atlas['@id'], self.JSON_ATLAS_ID)
        self.assertTrue(len(json_atlas['spaces']) == 1)
        self.assertTrue(len(json_atlas['parcellations']) == 2)

    def test_from_json_with_invalid_id(self):
        invalid_atlas_json = {
            "@id": "foo",
            "bar": "bar",
            "order": 1,
        }
        json_atlas = atlas.Atlas.from_json(invalid_atlas_json)
        self.assertEqual(json_atlas, invalid_atlas_json)

    def test_from_json_with_invalid_json(self):
        # Error handling for wrong json input
        pass

    def test_select_parcellation(self):
        selected_parcellation = self.atlas.selected_parcellation
        self.assertEqual(selected_parcellation, 'Julich-Brain Cytoarchitectonic Maps 2.5')

        new_parcellation = self.atlas.parcellations[2]
        self.atlas.select_parcellation(new_parcellation)

        self.assertEqual(self.atlas.selected_parcellation, new_parcellation)
        self.assertNotEqual(self.atlas.selected_parcellation, 'Julich-Brain Cytoarchitectonic Maps 2.5')

    def test_get_map(self):
        # test downloading map
        pass

    def test_get_mask(self):
        # test the binary mask for space
        pass

    def test_get_template(self):
        # test downloading template
        pass

    def test_select_region_from_instance(self):
        pass

    def test_select_region_from_key(self):
        pass

    def test_select_region_same_as_previous(self):
        pass

    def test_clear_selection(self):
        pass

    def test_region_selected(self):
        pass

    def test_coordinate_selected(self):
        pass

    def test_get_features(self):
        # more than one test needed
        # wrong modality, no region selected
        # GeneExpression
        pass

    def test_regionsprops(self):
        pass

    def test_regionprops_summarize(self):
        pass


if __name__ == "__main__":
    unittest.main()
