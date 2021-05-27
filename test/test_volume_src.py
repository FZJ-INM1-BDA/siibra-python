import unittest

from siibra import volumesrc

class TestVolumeSrc(unittest.TestCase):

    def test_volume_src_init(self):
        id='test_id'
        name='test_name'
        v_type='ng_precomputed'
        url='http://localhost/test'
        volume_src = volumesrc.VolumeSrc(id, name, url)
        self.assertIsNotNone(volume_src)
        self.assertEqual(volume_src.get_url(), url)

    def test_volume_from_valid_json(self):
        v_json = {
            '@id': 'json_id',
            '@type': 'fzj/tmp/volume_type/v0.0.1',
            'name': 'json_name',
            'volume_type': 'nii',
            'url':'http://localhost/test'
        }
        output = volumesrc.from_json(v_json)
        self.assertIsInstance(output, volumesrc.VolumeSrc)

    def test_volume_from_invalid_json(self):
        v_invalid_json = {
            '@id': 'json_id',
            '@type': 'fzj/tmp/not_volume_type/v0.0.1',
            'name': 'json_name',
            'volume_type': 'json_volume_type',
            'url':'http://localhost/test'
        }
        with self.assertRaises(NotImplementedError):
            output = volumesrc.from_json(v_invalid_json)

if __name__ == "__main__":
    unittest.main()
