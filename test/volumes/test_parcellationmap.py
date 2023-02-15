import unittest
from unittest.mock import patch, MagicMock
from siibra.volumes.parcellationmap import Map, space, parcellation, MapType, MapIndex, ExcessiveArgumentException, InsufficientArgumentException, ConflictingArgumentException, NonUniqueIndexError
from siibra.commons import Species
from siibra.core.region import Region
from uuid import uuid4
from parameterized import parameterized
import random
from itertools import product
import inspect


class DummyCls:
    def fetch(self):
        raise NotImplementedError


possible_indicies = list(
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
    )
)

possible_regions = ("region-foo", None)

volume_fetch_param = {
    "resolution_mm": (1e-6, ),
    "format": ("foo", ),
    "voi": (DummyCls(), ),
    "variant": ("hello", ),
    "fragment": ("foo-fragment", None),
    "index": [*possible_indicies, None],
    "region": [*possible_regions, None],
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
        return Map(
            identifier=str(uuid4()),
            name="map-name",
            space_spec=space_spec,
            parcellation_spec=parcellation_spec,
            indices=indices,
            volumes=volumes
        )

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
        ({"@id": "foo"}, "foo", True, True),
        ({"name": "bar"}, "bar", True, False),
        ({"buzz": "bay"}, None, False, None),
        ({}, None, False, None),
    ])
    def test_space(self, set_space_spec, called_arg, called_get_instance, return_space_flag):

        self.map = TestMap.get_instance(space_spec=set_space_spec)

        with patch.object(space.Space, 'get_instance') as mock_get_instance:
            return_space = space.Space(None, "Returned Space", species=Species.HOMO_SAPIENS) if return_space_flag else None
            mock_get_instance.return_value = return_space

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
        ({"@id": "foo"}, "foo", True, True),
        ({"name": "bar"}, "bar", True, False),
        ({"buzz": "bay"}, None, False, None),
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
        }, MapType.STATISTICAL, None),
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
                self.map = TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
                self.map.maptype
        else:
            self.map = TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
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
        self.map = TestMap.get_instance(indices=indices, volumes=[DummyCls(), DummyCls()])
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
            volumes,
            None
        )
        for args in map(
            lambda b: [b],
            [*possible_regions, *possible_indicies]
        )
        for kwargs in get_permutations_kwargs_fetch_params()
        for volumes in [
            [],
            [DummyCls()],
            [DummyCls() for _ in range(5)],
        ]
    ])
    def test_fetch(self, args, kwargs, indices, mock_fetch_idx, volumes, expected_meshindex):

        region_index_arg = args[0] if len(args) > 0 else None
        region_kwarg = kwargs.get("region")
        index_kwarg = kwargs.get("index")

        expected_error = None

        len_arg = len([arg for arg in [region_index_arg, region_kwarg, index_kwarg] if arg is not None])

        index = index_kwarg or (region_index_arg if isinstance(region_index_arg, MapIndex) else None)
        region = region_kwarg or (region_index_arg if isinstance(region_index_arg, (str, Region)) else None)

        if len_arg == 0 and len(volumes) != 1:
            expected_error = InsufficientArgumentException
        elif len_arg > 1:
            expected_error = ExcessiveArgumentException
        elif (
            index is not None
            and index.fragment is not None
            and kwargs.get("fragment") is not None
            and index.fragment != kwargs.get("fragment")
        ):
            expected_error = ConflictingArgumentException
        elif len(volumes) == 0:
            expected_error = IndexError

        mock_map_index = MapIndex(0, 0)
        if index is None:
            # default mapindex used by parcellationmap if everything is missing
            index = MapIndex(volume=0, label=None) if region is None else mock_map_index

        with patch.object(Map, 'get_index', return_value=mock_map_index) as get_index_mock:
            if expected_error is not None:
                assert inspect.isclass(expected_error)
                try:
                    with self.assertRaises(expected_exception=expected_error):
                        self.map.fetch(*args, **kwargs)
                        if len_arg <= 1 and region is not None:
                            get_index_mock.assert_called_once_with(region)
                except Exception as err:
                    raise err
                return

            assert index is not None
            expected_fragment_kwarg = kwargs.get("fragment") or index.fragment

            expected_label_kwarg = None
            if index is not None:
                expected_label_kwarg = index.label

            selected_volume = volumes[mock_fetch_idx]
            selected_volume.fetch = MagicMock()
            selected_volume.fetch.return_value = DummyCls()
            self.map = TestMap.get_instance(indices=indices, volumes=volumes)

            actual_fetch_result = self.map.fetch(*args, **kwargs)
            if region is not None:
                get_index_mock.assert_called_once_with(region)

            fetch_call_params = {
                key: kwargs.get(key)
                for key in volume_fetch_param
                if key not in ("region", "index")
            }
            fetch_call_params.pop("fragment", None)
            try:
                selected_volume.fetch.assert_called_once_with(
                    **fetch_call_params,
                    fragment=expected_fragment_kwarg,
                    label=expected_label_kwarg
                )
            except AssertionError:
                import pdb
                pdb.set_trace()
            self.assertIs(actual_fetch_result, selected_volume.fetch.return_value)


    @parameterized.expand(
        product(
            ["foo", Region("")],
            [
                ({}, NonUniqueIndexError),
                ({"foo": "bar", "baz": "world"}, NonUniqueIndexError),
                ({"foo": "baz"}, None)
            ]
        )
    )
    def test_get_index(self, region, find_index_stubs):
        return_find_indicies, ErrorCls = find_index_stubs
        with patch.object(Map, "find_indices", return_value=return_find_indicies) as mock:
            if inspect.isclass(ErrorCls):
                with self.assertRaises(ErrorCls):
                    self.map.get_index(region)
            else:
                return_val = self.map.get_index(region)
                self.assertIs(return_val, list(return_find_indicies.keys())[0])
            
            mock.assert_called_once_with(region)
