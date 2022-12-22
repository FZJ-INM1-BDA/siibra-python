import unittest
from unittest.mock import patch, MagicMock
from siibra.volumes.parcellationmap import Map, space, parcellation, MapType, MapIndex
from uuid import uuid4
from parameterized import parameterized
import random

class DummyCls:
    def fetch(self):
        raise NotImplementedError

volume_fetch_param = {
    "resolution_mm": (1e-6, 2e-3),
    "format": ("foo", "bar", "buzz"),
    "voi": ( DummyCls(), DummyCls() ),
    "variant": ("hello", "world"),
    "fragment": ("foo-fragment", None)
}

def get_randomised_fetch_params():
    return {key: random.sample(value, 1)[0] for key, value in volume_fetch_param.items()}

class TestMap(unittest.TestCase):
    @staticmethod
    def get_instance(space_spec={}, parcellation_spec={}, indices={}, volumes=[]):
        return Map(identifier=str(uuid4()),
            name="map-name",
            space_spec=space_spec,
            parcellation_spec=parcellation_spec,
            indices=indices,
            volumes=volumes)

    @classmethod
    def setUpClass(cls) -> None:
        cls.map = TestMap.get_instance()

    def test_init(self):
        assert self.map is not None
    
    def test_find_indicies(self):
        self.map = TestMap.get_instance(indices={
            'foo-bar': [{
                'volume': 0
            }]
        }, volumes=[DummyCls()])

    @parameterized.expand([
        ({ "@id": "foo" }, "foo", True, True),
        ({ "name": "bar" }, "bar", True, False),
        ({ "buzz": "bay" }, None, False, None),
        ({}, None, False, None),
    ])
    def test_space(self, set_space_spec, called_arg, called_get_instance, return_space_flag):
        
        self.map = TestMap.get_instance(space_spec=set_space_spec)

        with patch.object(space.Space, 'get_instance') as mock_get_instance:
            return_space = space.Space(None, "Returned Space") if return_space_flag else None
            mock_get_instance.return_value=return_space

            actual_returned_space = self.map.space

            if called_get_instance:
                mock_get_instance.assert_called_once_with(called_arg)
            else:
                mock_get_instance.assert_not_called()
            
            if not called_get_instance:
                assert actual_returned_space.name == "Unspecified space"
            else:
                assert actual_returned_space is return_space
    
    @parameterized.expand([
        ({ "@id": "foo" }, "foo", True, True),
        ({ "name": "bar" }, "bar", True, False),
        ({ "buzz": "bay" }, None, False, None),
        ({}, None, False, None),
    ])
    def test_parcellation(self, set_parc_spec, called_arg, called_get_instance, return_parc_flag):
        
        self.map = TestMap.get_instance(parcellation_spec=set_parc_spec)

        with patch.object(parcellation.Parcellation, 'get_instance') as mock_get_instance:
            return_value = DummyCls() if return_parc_flag else None
            mock_get_instance.return_value = return_value

            actual_returned_parcellation = self.map.parcellation

            if called_get_instance:
                mock_get_instance.assert_called_once_with(called_arg)
            else:
                mock_get_instance.assert_not_called()
            
            if not called_get_instance:
                assert actual_returned_parcellation is None
            else:
                assert actual_returned_parcellation is return_value

    @parameterized.expand([
        ({
            "foo": [{
                "volume": 0,
                "label": 1
            }],
            "bar": [{
                "volume": 0,
                "label": 2
            }]
        }, {1, 2}),
        ({
            "foo": [{
                "volume": 0
            }],
            "bar": [{
                "volume": 1
            }]
        }, {None})
    ])
    def test_labels(self, indices, compare_set):
        self.map = TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
        self.assertSetEqual(self.map.labels, compare_set)

    @parameterized.expand([
        ({
            "foo": [{
                "volume": 0,
                "label": 1
            }],
            "bar": [{
                "volume": 0,
                "label": 2
            }]
        }, MapType.LABELLED),
        ({
            "foo": [{
                "volume": 0
            }],
            "bar": [{
                "volume": 1
            }]
        }, MapType.CONTINUOUS),
        ({
            "foo": [{
                "volume": 0,
                "label": 0.125
            }]
        }, None),
        ({
            "foo": [{
                "volume": 0,
                "label": "foo"
            }]
        }, None)
    ])
    def test_maptype(self, indices, maptype):
        self.map=TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
        if maptype is None:
            with self.assertRaises(RuntimeError):
                self.map.maptype
        else:
            self.assertIs(self.map.maptype, maptype)

    @parameterized.expand([
        ({}, []),
        ({
            "foo": [{
                "volume": 0,
                "label": 1
            }],
            "bar": [{
                "volume": 0,
                "label": 2
            }]
        }, ["foo", "bar"]),
    ])
    def test_regions(self, indices, expected_regions):
        self.map=TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
        self.assertListEqual(self.map.regions, expected_regions)

    @parameterized.expand([
        (("region-foo",), { **get_randomised_fetch_params() }, {
            "foo": [{
                "volume": 0,
                "label": 1
            }],
            "bar": [{
                "volume": 0,
                "label": 2
            }]
        }, 0, None)
    ])
    def test_fetch(self, args, kwargs, indices, mock_fetch_idx, expected_meshindex):

        with patch.object(Map, 'get_index') as get_index_mock:
            get_index_mock.return_value = MapIndex(0, 0)
            
            volumes = [DummyCls() for _ in range(5)]
            selected_volume = volumes[mock_fetch_idx]
            selected_volume.fetch = MagicMock()
            selected_volume.fetch.return_value = DummyCls()
            self.map=TestMap.get_instance(indices=indices, volumes=volumes)
            
            actual_fetch_result = self.map.fetch(*args, **kwargs)
            fetch_call_params = {
                key: kwargs.get(key) for key in volume_fetch_param
            }
            selected_volume.fetch.assert_called_once_with(
                **fetch_call_params,
                mapindex=get_index_mock.return_value
            )
            self.assertIs(actual_fetch_result, selected_volume.fetch.return_value)

