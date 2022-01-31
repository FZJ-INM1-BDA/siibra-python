import unittest
import pytest

from siibra.volumes import VolumeSrc, RemoteNiftiVolume
from siibra import spaces, parcellations


class TestVolumeSrc(unittest.TestCase):
    def test_volumes_init(self):
        id = "test_id"
        name = "test_name"
        url = "http://localhost/test"
        volume_src = VolumeSrc(id, name, url, spaces[0])
        self.assertIsNotNone(volume_src)
        self.assertEqual(volume_src.url, url)

    def test_volume_from_valid_json(self):
        v_json = {
            "@id": "json_id",
            "@type": "fzj/tmp/volume_type/v0.0.1",
            "name": "json_name",
            "space_id": spaces[0],
            "volume_type": "nii",
            "url": "http://localhost/test",
        }
        output = VolumeSrc._from_json(v_json)
        self.assertIsInstance(output, VolumeSrc)
        self.assertIsInstance(output, RemoteNiftiVolume)

    def test_volume_from_invalid_json(self):
        v_invalid_json = {
            "@id": "json_id",
            "@type": "fzj/tmp/not_volume_type/v0.0.1",
            "name": "json_name",
            "volume_type": "json_volume_type",
            "url": "http://localhost/test",
        }
        with self.assertRaises(NotImplementedError):
            VolumeSrc._from_json(v_invalid_json)


space_volumes = [ volume
                for space in spaces
                for volume in space.volumes]


@pytest.mark.parametrize("volume", space_volumes)
def test_space_volumes(volume: VolumeSrc):
    volume.to_model()


parcs_volumes = [volume
                for parc in parcellations
                for volume in parc.volumes]


@pytest.mark.parametrize("volume", parcs_volumes)
def test_parc_volumes(volume: VolumeSrc):
    volume.to_model()


region_volmes = [volume
                for parc in parcellations
                for region in parc
                for volume in region.volumes]

@pytest.mark.parametrize("volume", region_volmes)
def test_region_volumes(volume: VolumeSrc):
    volume.to_model()


if __name__ == "__main__":
    unittest.main()
