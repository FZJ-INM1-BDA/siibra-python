import unittest
from unittest.mock import patch, MagicMock
from siibra.volumes.parcellationmap import Map, space, parcellation, MapType, MapIndex, ExcessiveArgumentException, InsufficientArgumentException, ConflictingArgumentException
from siibra._commons import Species
from uuid import uuid4
from parameterized import parameterized
import random
from itertools import product

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

def get_randomised_kwargs_fetch_params():
    return {key: random.sample(value, 1)[0] for key, value in volume_fetch_param.items()}

def get_permutations_kwargs_fetch_params():
    
    for zipped_value in product(*[
        product(values)
        for values in volume_fetch_param.values()
    ]):
        yield {
            key: value[0]
            for key, value in zip(volume_fetch_param.keys(), zipped_value)
        }

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
            return_space = space.Space(None, "Returned Space", species=Species.HOMO_SAPIENS) if return_space_flag else None
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
        }, MapType.LABELLED, None),
        ({
            "foo": [{
                "volume": 0
            }],
            "bar": [{
                "volume": 1
            }]
        }, MapType.CONTINUOUS, None),
        ({
            "foo": [{
                "volume": 0,
                "label": 0.125
            }]
        }, None, AssertionError),
        ({
            "foo": [{
                "volume": 0,
                "label": "foo"
            }]
        }, None, AssertionError)
    ])
    def test_maptype(self, indices, maptype, Error):
        if Error is not None:
            with self.assertRaises(Error):
                self.map=TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
                self.map.maptype
        else:
            self.map=TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
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
        (
            # args
            args,
            # kwargs
            kwargs,
            {
                "foo": [{
                    "volume": 0,
                    "label": 1
                }],
                "bar": [{
                    "volume": 0,
                    "label": 2
                }]
            },
            0,
            # volumes
            volumes
            ,
            None
        )
        for args in product(
            ("region-foo", None),
            # all possible permutations of map
            filter(
                lambda idx: idx is not None, 
                map(
                    lambda args: MapIndex(*args) if args[0] is not None or args[1] is not None else None,
                    product(
                        (0, None),
                        (0, None),
                        ('bla-fragment', 'foo-fragment', None)
                    )
                )
            ),
        )
        for kwargs in get_permutations_kwargs_fetch_params()
        for volumes in [
            [],
            [DummyCls()],
            [DummyCls() for _ in range(5)],
        ]
    ])
    def test_fetch(self, args, kwargs, indices, mock_fetch_idx, volumes, expected_meshindex):

        region, index = args
        
        expected_error = None
        if region is None and index is None and len(volumes) != 1:
            expected_error = InsufficientArgumentException
        elif len([True for _ in [region, index] if _ is not None]) != 1:
            expected_error = ExcessiveArgumentException
        elif all([
            index is not None,
            index.fragment is not None,
            kwargs.get("fragment") is not None,
            index.fragment != kwargs.get("fragment"),
        ]):
            expected_error = ConflictingArgumentException
        elif len(volumes) == 0:
            expected_error = IndexError


        expected_fragment_kwarg = kwargs.get('fragment') or index.fragment

        mock_map_index = MapIndex(0, 0)
        expected_label_kwarg = mock_map_index.label if region is not None else index.label

        with patch.object(Map, 'get_index') as get_index_mock:
            get_index_mock.return_value = mock_map_index
            
            if expected_error is not None:
                try:
                    with self.assertRaises(expected_exception=expected_error):
                        self.map.fetch(*args, **kwargs)
                        if region is not None:
                            get_index_mock.assert_called_once_with(region)

                except Exception as err:
                    import pdb
                    pdb.set_trace()
                return

            selected_volume = volumes[mock_fetch_idx]
            selected_volume.fetch = MagicMock()
            selected_volume.fetch.return_value = DummyCls()
            self.map=TestMap.get_instance(indices=indices, volumes=volumes)

            actual_fetch_result = self.map.fetch(*args, **kwargs)
            if region is not None:
                get_index_mock.assert_called_once_with(region)

            fetch_call_params = {
                key: kwargs.get(key) for key in volume_fetch_param
            }
            fetch_call_params.pop("fragment", None)
            selected_volume.fetch.assert_called_once_with(
                **fetch_call_params,
                fragment=expected_fragment_kwarg,
                label=expected_label_kwarg
            )
            self.assertIs(actual_fetch_result, selected_volume.fetch.return_value)
