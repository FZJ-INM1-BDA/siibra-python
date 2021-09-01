import os
import unittest
import siibra
from siibra import atlases, parcellations, modalities
from siibra.core import atlas
from test.get_token import get_token

token = get_token()
os.environ['HBP_AUTH_TOKEN'] = token["access_token"]


class TestAtlas(unittest.TestCase):

    JSON_ATLAS_ID = "json_1337"
    JSON_ATLAS_NAME = "JSON Human Atlas"
    atlas_as_json = {
        "@type": "juelich/iav/atlas/v1.0.0",
        "@id": JSON_ATLAS_ID,
        "name": JSON_ATLAS_NAME,
        "order": 1,
        "spaces": [
            "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2",
        ],
        "parcellations": [
            "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290",
            "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579",
        ]
    }

    @classmethod
    def tearDownClass(cls):
        cls.parcellation = cls.atlas.get_parcellation('2.9')

    @classmethod
    def setUpClass(cls):
        cls.atlas = atlases.MULTILEVEL_HUMAN_ATLAS
        cls.ATLAS_NAME = 'Multilevel Human Atlas'

    def test_atlas_init(self):
        a = atlas.Atlas('juelich/iav/atlas/v1.0.0/1', self.ATLAS_NAME)
        self.assertEqual(a.name, self.ATLAS_NAME)
        self.assertEqual(a.key, 'MULTILEVEL_HUMAN_ATLAS')
        self.assertEqual(a.id, 'juelich/iav/atlas/v1.0.0/1')

        # on init, parcellations and spaces are empty lists
        self.assertTrue(len(a._parcellations) == 0)
        self.assertTrue(len(a.spaces) == 0)

    def test_parcellations(self):
        parcellations = self.atlas.parcellations
        self.assertTrue(len(parcellations) >= 11)
        parcellation = self.atlas.get_parcellation(self.atlas.parcellations[0])
        self.assertIsNotNone(parcellation)

    def test_spaces(self):
        spaces = self.atlas.spaces
        self.assertTrue(len(spaces) == 4)

    def test_to_string(self):
        self.assertTrue(self.ATLAS_NAME in str(self.atlas))

    def test__from_json(self):
        json_atlas = atlas.Atlas._from_json(self.atlas_as_json)

        self.assertTrue(type(json_atlas) is atlas.Atlas)

        self.assertEqual(json_atlas.name, self.JSON_ATLAS_NAME)
        self.assertEqual(json_atlas.id, self.JSON_ATLAS_ID)
        self.assertTrue(len(json_atlas.spaces) == 1)
        self.assertTrue(len(json_atlas.parcellations) == 2)

    def test__from_json_with_invalid_id(self):
        invalid_atlas_json = {
            "@id": "foo",
            "bar": "bar",
            "order": 1,
        }
        with self.assertRaises(ValueError):
            atlas.Atlas._from_json(invalid_atlas_json)

    def test__from_json_with_invalid_json(self):
        # Error handling for wrong json input
        pass

    def test_select_parcellation(self):
        selected_parcellation = self.atlas.get_parcellation()
        self.assertEqual(selected_parcellation, 'Julich-Brain Cytoarchitectonic Maps 2.9')

        new_parcellation = self.atlas.parcellations[2]
        selected_parcellation = self.atlas.get_parcellation(new_parcellation)

        self.assertEqual(selected_parcellation, new_parcellation)
        self.assertNotEqual(selected_parcellation, 'Julich-Brain Cytoarchitectonic Maps 2.9')

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

    def test_get_features_ebrains_features(self):

        region = self.atlas.get_region('hoc1 left')
        features = siibra.get_features(region, modalities.EbrainsRegionalDataset)
        assert(len(features) > 0)

    def test_get_features_connectivity_profile_filter_by_sel_parc(self):

        expected_p_id = 'minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290'
        region = self.atlas.get_region(
            parcellation=parcellations[expected_p_id],
            region='hoc1 left')
        conns = siibra.get_features(region, modalities.ConnectivityProfile)
        assert(len(conns) > 0)
        assert(all([region.parcellation in conn._matrix.parcellations for conn in conns]))
        assert(all([region.matches(conn.regionspec) for conn in conns]))

    def test_regionsprops(self):
        pass

    def test_regionprops_summarize(self):
        pass


if __name__ == "__main__":
    unittest.main()
