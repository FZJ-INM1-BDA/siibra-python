import unittest
import siibra
from siibra.core import Parcellation

class TestParcellationVersion(unittest.TestCase):
    correct_json={
        'name': 'foobar',
        'collectionName': 'foobar-collection',
        '@prev': 'foobar-prev',
        '@next': 'foobar-next',
        'deprecated': False,
    }
    def test__from_json(self):

        ver=siibra.core.parcellation.ParcellationVersion._from_json(self.correct_json)
        self.assertTrue(ver.deprecated == self.correct_json['deprecated'])
        self.assertTrue(ver.name == self.correct_json['name'])
        self.assertTrue(ver.collection == self.correct_json['collectionName'])
        self.assertTrue(ver.prev_id == self.correct_json['@prev'])
        self.assertTrue(ver.next_id == self.correct_json['@next'])
        # TODO test prev/next
        
class TestParcellation(unittest.TestCase):

    correct_json={
        '@id': 'id-foo',
        '@type': 'minds/core/parcellationatlas/v1.0.0',
        'shortName': 'id-foo-shortname',
        'name':'fooparc',
        'regions': []
    }

    correct_json_no_type={
        **correct_json,
        '@type': 'foo-bar'
    }

    def test_from_json_malformed(self):
        self.assertRaises(AssertionError, lambda: Parcellation._from_json(self.correct_json_no_type))
    
    def test_from_json(self):
        parc = Parcellation._from_json(self.correct_json)
        assert isinstance(parc, Parcellation)

    def test_find_regions_ranks_result(self):
        updated_json = {
            **self.correct_json,
            'regions': [{
                'name': 'foo bar',
                'children': [{
                    'name': 'foo'
                }]
            }]
        }
        parc = Parcellation._from_json(updated_json)
        regions = parc.find_regions('foo')
        assert len(regions) == 3
        assert regions[0].name == 'foo'


if __name__ == "__main__":
    unittest.main()
