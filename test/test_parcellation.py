import unittest
import siibra

class TestParcellationVersion(unittest.TestCase):
    correct_json={
        'name': 'foobar',
        'collectionName': 'foobar-collection',
        '@prev': 'foobar-prev',
        '@next': 'foobar-next',
        'deprecated': False,
    }
    def test_from_json(self):

        ver=siibra.parcellation.ParcellationVersion.from_json(self.correct_json)
        self.assertTrue(ver.deprecated == self.correct_json['deprecated'])
        self.assertTrue(ver.name == self.correct_json['name'])
        self.assertTrue(ver.collection == self.correct_json['collectionName'])
        self.assertTrue(ver.prev_id == self.correct_json['@prev'])
        self.assertTrue(ver.next_id == self.correct_json['@next'])
        # TODO test prev/next
        


if __name__ == "__main__":
    unittest.main()
