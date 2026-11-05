import gc
import unittest
from unittest.mock import patch
import numpy as np
from siibra.volumes.volume import Volume
from siibra.volumes.providers.nifti import NiftiProvider
from siibra.core import space
from siibra.volumes.providers.provider import VolumeProvider
from parameterized import parameterized


class DummyVolumeProvider(VolumeProvider, srctype="foo-bar"):

    def fetch(self, *args, **kwargs):
        pass

    @property
    def _url(self):
        return {}

    def get_boundingbox(self, clip=True, background=0):
        return None


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
        self.assertSetEqual(self.volume.formats, {'foo-bar'})

    @parameterized.expand([
        ({"@id": "foo"}, "foo", True, True),
        ({"name": "bar"}, "bar", True, False),
        ({"buzz": "bay"}, None, False, None),
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


class TestVolumeFetchCache(unittest.TestCase):
    """The fetch cache must not return the result of another volume with the same hash."""

    @staticmethod
    def in_memory_volume(shape) -> Volume:
        data = np.ones(shape, dtype="float32")
        return Volume(space_spec={}, providers=[NiftiProvider((data, np.eye(4)))])

    def setUp(self) -> None:
        cache_patcher = patch.object(Volume, "_FETCH_CACHE", {})
        cache_patcher.start()
        self.addCleanup(cache_patcher.stop)
        # emulate a volume created at the memory address of a garbage-collected one
        hash_patcher = patch.object(Volume, "__hash__", lambda self: 0)
        hash_patcher.start()
        self.addCleanup(hash_patcher.stop)

    def test_result_of_collected_volume_is_not_returned(self):
        first = self.in_memory_volume((2, 3, 4, 5))
        self.assertEqual(first.fetch().shape, (2, 3, 4, 5))
        del first
        gc.collect()
        second = self.in_memory_volume((2, 3, 4))
        self.assertEqual(second.fetch().shape, (2, 3, 4))

    def test_result_of_other_live_volume_is_not_returned(self):
        first = self.in_memory_volume((2, 3, 4, 5))
        second = self.in_memory_volume((2, 3, 4))
        self.assertEqual(first.fetch().shape, (2, 3, 4, 5))
        self.assertEqual(second.fetch().shape, (2, 3, 4))

    def test_repeated_fetch_is_cached(self):
        volume = self.in_memory_volume((2, 3, 4))
        self.assertIs(volume.fetch(), volume.fetch())
