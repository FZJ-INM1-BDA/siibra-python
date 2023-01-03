import unittest
import pytest
from unittest.mock import patch
from siibra._commons import MapType
from siibra.volumes.volume import Volume, VolumeProvider, space
from parameterized import parameterized

class DummyVolumeProvider(VolumeProvider, srctype="foo-bar"):
    def fetch(self, *args, **kwargs): pass

    @property
    def _url(self):
        return {}

class TestVolumeProvider(unittest.TestCase):

    @staticmethod
    def get_instance():
        return DummyVolumeProvider()
    
    def test_volume_srctype(self):
        self.assertEqual(DummyVolumeProvider.srctype, "foo-bar")

class TestVolume(unittest.TestCase):

    @staticmethod
    def get_instance(space_spec={}):
        return Volume(space_spec=space_spec, providers=[TestVolumeProvider.get_instance()], name="test-volume")

    @classmethod
    def setUpClass(cls) -> None:
        cls.volume = TestVolume.get_instance()

    def test_init(self):
        self.assertIsNotNone(self.volume)
    
    def test_formats(self):
        self.assertSetEqual(self.volume.formats, {'foo-bar', 'image'})
    
    @parameterized.expand([
        ({ "@id": "foo" }, "foo", True, True),
        ({ "name": "bar" }, "bar", True, False),
        ({ "buzz": "bay" }, None, False, None),
        ({}, None, False, None),
    ])
    def test_space(self, set_space_spec, called_arg, called_get_instance, returned_space):
        
        self.volume = TestVolume.get_instance(space_spec=set_space_spec)

        with patch.object(space.Space, 'get_instance') as mock_get_instance:
            return_space = space.Space(None, "Returned Space", space.Species.UNSPECIFIED_SPECIES) if returned_space else None
            mock_get_instance.return_value = return_space

            actual_returned_space = self.volume.space

            if called_get_instance:
                mock_get_instance.assert_called_once_with(called_arg)
            else:
                mock_get_instance.assert_not_called()
            
            if not called_get_instance:
                assert actual_returned_space.name == "Unspecified space"
            else:
                assert actual_returned_space is return_space

    def test_fetch(self):
        # TODO add after tests for boudningbox are added
        pass


# TODO move to int test
# fetch_ng_volume_fetchable_params = [
#     ("ID", "NAME", "https://neuroglancer.humanbrainproject.eu/precomputed/data-repo-ng-bot/20210927-waxholm-v4/precomputed/segmentations/WHS_SD_rat_atlas_v4", None, None)
# ]


# @pytest.mark.parametrize("identifier,name,url,space,detail", fetch_ng_volume_fetchable_params)
# def test_ng_volume(identifier, name, url, space, detail):
#     vol = NeuroglancerVolumeFetcher(identifier, name, url, space, detail)
#     vol.fetch()


# volume_map_types = [
#     ("difumo 64", NeuroglancerVolumeFetcher, 0, MapType.LABELLED),
#     ("difumo 128", NeuroglancerVolumeFetcher, 0, MapType.LABELLED),
#     ("difumo 256", NeuroglancerVolumeFetcher, 0, MapType.LABELLED),
#     ("difumo 512", NeuroglancerVolumeFetcher, 0, MapType.LABELLED),
#     ("difumo 1024", NeuroglancerVolumeFetcher, 0, MapType.LABELLED),
# ]


# @pytest.mark.parametrize("parc_id,volume_cls,volume_index,map_type", volume_map_types)
# def test_volume_map_types(parc_id, volume_cls, volume_index, map_type):
#     parc = parcellations[parc_id]
#     v: VolumeSrc = [v for v in parc.volumes if isinstance(v, volume_cls)][volume_index]
#     assert v.map_type is map_type


# if __name__ == "__main__":
#     unittest.main()
