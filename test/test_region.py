import unittest
from unittest.mock import MagicMock, patch

from siibra import parcellations, ebrains, spaces, retrieval
from siibra.region import Region


class TestRegions(unittest.TestCase):

    region_name = 'Interposed Nucleus (Cerebellum) - left hemisphere'
    kg_id = '658a7f71-1b94-4f4a-8f15-726043bbb52a'
    parentname = 'region_parent'

    definition = {
        'name': region_name,
        'rgb': [170, 29, 10],
        'labelIndex': 251,
        'ngId': 'jubrain mni152 v18 left',
        'children': [],
        'position': [-9205882, -57128342, -32224599],
        'originDatasets': [ {
            'kgId': kg_id,
            'kgSchema': 'minds/core/dataset/v1.0.0',
            'filename': 'Interposed Nucleus (Cerebellum) [v6.2, ICBM 2009c Asymmetric, left hemisphere]'
        }]
    }

    parent_definition = {
        'name': parentname,
        'rgb': [170, 29, 10],
        'labelIndex': 251,
        'ngId': 'jubrain mni152 v18 left',
        'children': [],
        'position': [-9205882, -57128342, -32224599],
        'originDatasets': [ {
            'kgId': kg_id,
            'kgSchema': 'minds/core/dataset/v1.0.0',
            'filename': 'Interposed Nucleus (Cerebellum) [v6.2, ICBM 2009c Asymmetric, left hemisphere]'
        }]
    }
    parent_region = None
    child_region = None

    @classmethod
    def setUpClass(cls):
        retrieval.download_file = MagicMock()
        retrieval.download_file.return_value = None

        cls.parent_region = Region.from_json(cls.parent_definition,parcellations[0])

        cls.child_region = Region.from_json(cls.definition,parcellations[0])
        cls.child_region.parent=cls.parent_region

    def test_regions_init(self):
        self.assertEqual(str(self.child_region), self.region_name)

    def test_related_ebrains_files(self):
        ebrains.execute_query_by_id = MagicMock()
        ebrains.execute_query_by_id.return_value = {
            'results': [
                {
                    'files': ['file1', 'file2']
                }
            ]
        }
        files = self.child_region._related_ebrains_files()
        ebrains.execute_query_by_id.assert_called_with(
            'minds',
            'core',
            'dataset',
            'v1.0.0',
            'brainscapes_files_in_dataset',
            params={'dataset': self.kg_id})

        self.assertTrue(len(files) == 2)

    def test_related_ebrains_files_with_no_origin_datasets(self):
        ebrains.execute_query_by_id = MagicMock()
        definition = {
            'name': self.region_name,
            'rgb': [170, 29, 10],
            'labelIndex': 251,
            'ngId': 'jubrain mni152 v18 left',
            'children': [],
            'position': [-9205882, -57128342, -32224599],
        }
        region = Region(definition, parcellations[0])
        files = region._related_ebrains_files()
        ebrains.execute_query_by_id.assert_not_called()
        self.assertTrue(len(files) == 0)

    def test_related_ebrains_files_with_no_kg_id(self):
        ebrains.execute_query_by_id = MagicMock()
        self.child_region.attrs['originDatasets'] = [{
            'kgSchema': 'minds/core/dataset/v1.0.0',
            'filename': 'Interposed Nucleus (Cerebellum) [v6.2, ICBM 2009c Asymmetric, left hemisphere]'
        }]
        files = self.child_region._related_ebrains_files()
        ebrains.execute_query_by_id.assert_not_called()
        self.assertTrue(len(files) == 0)

    def test_related_ebrains_files_with_empty_origin_datasets(self):
        ebrains.execute_query_by_id = MagicMock()
        self.child_region.attrs['originDatasets'] = []
        files = self.child_region._related_ebrains_files()
        ebrains.execute_query_by_id.assert_not_called()
        self.assertTrue(len(files) == 0)

    def test_has_no_parent(self):
        self.assertFalse(self.parent_region.has_parent(self.parentname))

    def test_has_parent(self):
        self.assertTrue(self.child_region.has_parent(self.parentname))

    def test_includes_region_true(self):
        self.parent_region.children = [self.child_region]
        self.assertTrue(self.parent_region.includes(self.child_region))

    def test_includes_region_false(self):
        self.parent_region.children = []
        self.assertFalse(self.parent_region.includes(self.child_region))

    def test_includes_region_self(self):
        self.assertTrue(self.parent_region.includes(self.parent_region))

    def test_find_child_region(self):
        regions = self.parent_region.find(self.region_name)
        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 1)
        self.assertEqual(next(iter(regions)), self.child_region)

    def test_find_child_no_result(self):
        regions = self.child_region.find(self.parentname)
        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 0)

    def test_matches_with_valid_string(self):
        self.assertTrue(self.child_region.matches('Interposed Nucleus'))

    def test_matches_with_invalid_string(self):
        self.assertFalse(self.child_region.matches('Area 51'))

    def test_matches_with_valid_region(self):
        self.assertTrue(self.child_region.matches(self.child_region))

    def test_matches_with_wrong_region(self):
        self.assertFalse(self.child_region.matches(self.parent_region))

    @patch('siibra.region.download_file')
    def test_regional_map_none(self, download_mock):
        self.assertIsNone(self.parent_region.get_regional_map(spaces[0]))
        download_mock.assert_not_called()

    @patch('siibra.region.download_file')
    def test_get_regional_map_wrong_space(self, download_mock):
        self.child_region.attrs['maps'] = {
            'spaceId': 'map_url'
        }
        self.assertIsNone(self.child_region.get_regional_map(spaces[0]))
        download_mock.assert_not_called()

    @patch('siibra.region.download_file')
    def test_get_regional_map_no_filename(self, download_mock):
        download_mock.return_value = None
        self.child_region.attrs['maps'] = {
            spaces[0].id: spaces[0].id
        }
        self.assertIsNone(self.child_region.get_regional_map(spaces[0]))
        download_mock.assert_called_with(spaces[0].id)


if __name__ == "__main__":
    unittest.main()
