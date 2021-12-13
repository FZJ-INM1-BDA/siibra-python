import unittest

from siibra.volumes import VolumeSrc
from siibra import spaces
from siibra.volumes.volume import RemoteNiftiVolume


class TestVolumeSrc(unittest.TestCase):

    def test_volume_from_valid_json(self):
        v_json = {
            "@id": "json_id",
            "@type": "fzj/tmp/volume_type/v0.0.1",
            "name": "json_name",
            "space_id": spaces[0],
            "volume_type": "nii",
            "url": "http://localhost/test",
        }
        output_array = VolumeSrc.parse_legacy(v_json)
        assert len(output_array) == 1
        assert all(isinstance(output, VolumeSrc) for output in output_array)
        # TODO this does not yet work
        # assert all(isinstance(output, RemoteNiftiVolume) for output in output_array)

    def test_volume_from_invalid_json(self):
        v_invalid_json = {
            "@id": "json_id",
            "@type": "fzj/tmp/not_volume_type/v0.0.1",
            "name": "json_name",
            "volume_type": "json_volume_type",
            "url": "http://localhost/test",
        }
        with self.assertRaises(NotImplementedError):
            VolumeSrc.parse_legacy(v_invalid_json)


if __name__ == "__main__":
    unittest.main()
