import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from siibra.volumes.parcellationmap import (
    Map, space, parcellation, MapType, MapIndex
)
from siibra.exceptions import (
    ExcessiveArgumentException, ConflictingArgumentException,
    NonUniqueIndexError, NoVolumeFound
)
from siibra.commons import Species
from siibra.core.region import Region
from uuid import uuid4
from parameterized import parameterized
import random
from itertools import product
import inspect
import numpy as np
from nibabel import Nifti1Image
from siibra.volumes.volume import Volume
from siibra.locations.point import Point
from siibra.locations.pointcloud import PointCloud
from siibra.volumes.sparsemap import SparseMap, SparseIndex


class DummyParc():
    def __init__(self, name: str) -> None:
        self.name = name


class DummyCls:
    """Stand-in for a Volume (and for other objects the tests only pass around)."""

    def __init__(self, formats=(), provides_image=False, provides_mesh=False):
        self._providers = {}
        self.formats = set(formats)
        self.provides_image = provides_image
        self.provides_mesh = provides_mesh

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
        parcnames = [
            "parc-name",
            "prerelease - parc-name",
            "[PRERELEASE] - parc-name"
        ]
        with patch.object(Map, "parcellation") as mock_parc:
            for parcname in parcnames:
                mock_parc.return_value = DummyParc(parcname)
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

        region_arg = args[0] if len(args) > 0 else None

        region_kwarg = kwargs.get("region")
        index_kwarg = kwargs.get("index")

        expected_error = None

        len_arg = len([arg for arg in [region_arg, region_kwarg, index_kwarg] if arg is not None])

        index = index_kwarg
        region = region_kwarg or (region_arg if isinstance(region_arg, (str, Region)) else None)
        if "region" in kwargs or isinstance(region_arg, MapIndex):
            expected_error = TypeError
        else:
            if len_arg == 0 and len(volumes) != 1:
                expected_error = NoVolumeFound
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
            except AssertionError as e:
                print(e)
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


class TestMapGuards(unittest.TestCase):
    """
    Operations that are defined only for certain types of map must fail loudly.

    These conditions are easy to invert when converting an assert into a raise,
    which silently turns the valid case into the error case.
    """

    @staticmethod
    def statistical_map():
        return TestMap.get_instance(
            indices={"foo": [{"volume": 0}], "bar": [{"volume": 1}]},
            volumes=[DummyCls(formats={"nii"}, provides_image=True) for _ in range(2)],
        )

    @staticmethod
    def labelled_map(formats={"nii"}, provides_image=True, provides_mesh=False, indices=None, n_volumes=1):
        return TestMap.get_instance(
            indices=indices or {"foo": [{"volume": 0, "label": 1}]},
            volumes=[
                DummyCls(formats=formats, provides_image=provides_image, provides_mesh=provides_mesh)
                for _ in range(n_volumes)
            ],
        )

    def test_compress_rejects_statistical_maps(self):
        with self.assertRaises(ValueError):
            self.statistical_map().compress()

    def test_compress_rejects_maps_with_nothing_to_merge(self):
        with self.assertRaises(RuntimeError):
            self.labelled_map().compress()

    def test_bids_lookup_table_rejects_statistical_maps(self):
        with self.assertRaises(NotImplementedError):
            self.statistical_map().to_BIDS_lookup_table()

    def test_nilearn_masker_rejects_statistical_maps(self):
        with self.assertRaises(NotImplementedError):
            self.statistical_map().as_nilearn_masker()

    def test_colorize_rejects_statistical_maps(self):
        with self.assertRaises(NotImplementedError):
            self.statistical_map().colorize({"foo": 1.0})

    def test_nilearn_masker_rejects_a_user_supplied_lut(self):
        with self.assertRaises(ValueError):
            self.labelled_map().as_nilearn_masker(lut="something")

    def test_surfaceimage_requires_a_gii_label_map(self):
        with self.assertRaises(ValueError):
            self.labelled_map(formats={"nii"})._as_surfaceimage()

    def test_surfaceimage_requires_a_single_volume(self):
        mp = self.labelled_map(
            formats={"gii-label"}, provides_image=False, provides_mesh=True,
            indices={"foo": [{"volume": 0, "label": 1}], "bar": [{"volume": 1, "label": 1}]},
            n_volumes=2,
        )
        with self.assertRaises(ValueError):
            mp._as_surfaceimage()


class TestMapFragments(unittest.TestCase):
    def test_fragments_are_sorted(self):
        """Relabelling and column order depend on a stable fragment order."""
        mp = TestMap.get_instance(
            indices={
                "foo": [{"volume": 0, "label": 1, "fragment": "right hemisphere"}],
                "bar": [{"volume": 0, "label": 1, "fragment": "left hemisphere"}],
            },
            volumes=[DummyCls(formats={"gii-label"}, provides_mesh=True)],
        )
        self.assertEqual(mp.fragments, ["left hemisphere", "right hemisphere"])


class TestMapAssignLowerThreshold(unittest.TestCase):
    """`lower_threshold` acts on map values (statistical maps) and is rejected for labelled maps (#717)."""

    SHAPE = (10, 10, 10)

    def setUp(self) -> None:
        self.space = MagicMock()
        space_patcher = patch.object(Map, "space", new_callable=PropertyMock, return_value=self.space)
        space_patcher.start()
        self.addCleanup(space_patcher.stop)

        # regional map of "foo": a block of 0.8 next to a block of 0.2, 64 voxels each
        self.foo_arr = np.zeros(self.SHAPE, dtype="float32")
        self.foo_arr[0:4, 0:4, 0:4] = 0.8
        self.foo_arr[4:8, 0:4, 0:4] = 0.2
        # regional map of "bar": disjoint from the input
        self.bar_arr = np.zeros(self.SHAPE, dtype="float32")
        self.bar_arr[9, 9, 9] = 0.9

        # input: one connected component of 256 voxels containing both blocks of "foo"
        query_arr = np.zeros(self.SHAPE, dtype="uint8")
        query_arr[0:8, 0:8, 0:4] = 1
        self.queryvolume = MagicMock(spec=Volume)
        self.queryvolume.space = self.space
        self.queryvolume.fetch.return_value = Nifti1Image(query_arr, np.eye(4))

    def statistical_map(self) -> Map:
        mp = TestMap.get_instance(
            indices={"foo": [{"volume": 0}], "bar": [{"volume": 1}]},
            volumes=[DummyCls(formats={"nii"}, provides_image=True) for _ in range(2)],
        )
        images = {
            0: Nifti1Image(self.foo_arr, np.eye(4)),
            1: Nifti1Image(self.bar_arr, np.eye(4)),
        }
        fetch_patcher = patch.object(
            Map, "fetch", side_effect=lambda *args, index=None, **kwargs: images[index.volume]
        )
        fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)
        return mp

    def test_threshold_applies_to_map_values(self):
        mp = self.statistical_map()
        unthresholded = mp._assign(self.queryvolume, lower_threshold=0.0)
        thresholded = mp._assign(self.queryvolume, lower_threshold=0.5)
        self.assertEqual(len(unthresholded), 1)
        self.assertEqual(len(thresholded), 1)
        # only the 0.8 block remains: intersection 64, union 256.
        # A threshold on the score would have dropped this result (0.25 < 0.5).
        self.assertAlmostEqual(unthresholded[0].intersection_over_union, 128 / 256)
        self.assertAlmostEqual(thresholded[0].intersection_over_union, 64 / 256)

    def test_threshold_removing_all_overlap_yields_no_result(self):
        mp = self.statistical_map()
        self.assertEqual(mp._assign(self.queryvolume, lower_threshold=0.85), [])

    def test_regions_without_overlap_are_excluded(self):
        mp = self.statistical_map()
        assignments = mp._assign(self.queryvolume)
        self.assertEqual([a.volume for a in assignments], [0])

    def test_map_array_is_not_modified(self):
        mp = self.statistical_map()
        original = self.foo_arr.copy()
        mp._assign(self.queryvolume, lower_threshold=0.5)
        np.testing.assert_array_equal(self.foo_arr, original)

    @parameterized.expand(product((Point, PointCloud, Volume), (0.5, -1.0)))
    def test_labelled_map_rejects_nonzero_threshold(self, item_cls, threshold):
        mp = TestMapGuards.labelled_map()
        with patch.object(Map, "_assign_points") as mock_points, \
                patch.object(Map, "_assign_volume") as mock_volume:
            with self.assertRaises(ValueError):
                mp.assign(MagicMock(spec=item_cls), lower_threshold=threshold)
        mock_points.assert_not_called()
        mock_volume.assert_not_called()

    @parameterized.expand([(PointCloud, ), (Volume, )])
    def test_labelled_map_accepts_default_threshold(self, item_cls):
        mp = TestMapGuards.labelled_map()
        with patch.object(Map, "_assign_points", return_value=[]), \
                patch.object(Map, "_assign_volume", return_value=[]):
            result = mp.assign(MagicMock(spec=item_cls))
        self.assertEqual(len(result), 0)


class TestMapAssignScoreColumns(unittest.TestCase):
    """
    `Map` and `SparseMap` implement volume assignment separately. Both must
    give the same results: the same quantities in the score columns, the
    same handling of `lower_threshold` and of the query space (#717).

    The case is asymmetric so that swapped columns differ: the regional map
    (64 voxels of 0.5) is a strict subset of the input (128 voxels of 1).
    """

    SHAPE = (10, 10, 10)

    def setUp(self) -> None:
        self.space = MagicMock()
        for cls in (Map, SparseMap):
            patcher = patch.object(cls, "space", new_callable=PropertyMock, return_value=self.space)
            patcher.start()
            self.addCleanup(patcher.stop)
        region_patcher = patch.object(Map, "get_region", return_value="foo")
        region_patcher.start()
        self.addCleanup(region_patcher.stop)

        self.region_arr = np.zeros(self.SHAPE, dtype="float32")
        self.region_arr[0:4, 0:4, 0:4] = 0.5
        self.region_arrs = [self.region_arr]
        query_arr = np.zeros(self.SHAPE, dtype="uint8")
        query_arr[0:8, 0:4, 0:4] = 1
        self.queryvolume = MagicMock(spec=Volume)
        self.queryvolume.space = self.space
        self.queryvolume.fetch.return_value = Nifti1Image(query_arr, np.eye(4))

    def statistical_map(self) -> Map:
        mp = TestMap.get_instance(
            indices={f"region {i}": [{"volume": i}] for i in range(len(self.region_arrs))},
            volumes=[DummyCls(formats={"nii"}, provides_image=True) for _ in self.region_arrs],
        )
        images = [Nifti1Image(arr, np.eye(4)) for arr in self.region_arrs]
        fetch_patcher = patch.object(
            Map, "fetch", side_effect=lambda *args, index=None, **kwargs: images[index.volume]
        )
        fetch_patcher.start()
        self.addCleanup(fetch_patcher.stop)
        return mp

    def sparse_map(self) -> SparseMap:
        mp = SparseMap(
            identifier=str(uuid4()),
            name="sparse-map-name",
            space_spec={},
            parcellation_spec={},
            indices={f"region {i}": [{"volume": i}] for i in range(len(self.region_arrs))},
            volumes=[DummyCls(formats={"nii"}, provides_image=True) for _ in self.region_arrs],
        )
        spind = SparseIndex()
        for arr in self.region_arrs:
            spind.add_img(arr, np.eye(4))
        mp._sparse_index_cached = spind
        return mp

    @parameterized.expand([("statistical_map", ), ("sparse_map", )])
    def test_score_columns(self, factory):
        df = getattr(self, factory)().assign(self.queryvolume)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertAlmostEqual(row["intersection over union"], 64 / 128)
        self.assertAlmostEqual(row["map containedness"], 1.0)
        self.assertAlmostEqual(row["input containedness"], 64 / 128)
        # map values (0.5) averaged over the input: 64 * 0.5 / 128
        self.assertAlmostEqual(row["map weighted mean"], 0.25)
        # input values (1) averaged over the regional map
        self.assertAlmostEqual(row["input weighted mean"], 1.0)

    @parameterized.expand(product(("statistical_map", "sparse_map"), ((0.4, 1), (0.5, 0))))
    def test_only_map_values_above_threshold_count(self, factory, threshold_and_rows):
        threshold, expected_rows = threshold_and_rows
        df = getattr(self, factory)().assign(self.queryvolume, lower_threshold=threshold)
        self.assertEqual(len(df), expected_rows)

    @parameterized.expand(product(("statistical_map", "sparse_map"), (True, False)))
    def test_components_are_numbered_from_1(self, factory, split_components):
        df = getattr(self, factory)().assign(self.queryvolume, split_components=split_components)
        self.assertEqual(df["input structure"].tolist(), [1])

    @parameterized.expand([("statistical_map", ), ("sparse_map", )])
    def test_rows_are_ordered_by_component_then_map(self, factory):
        # two regional maps, each overlapped by both input components
        self.region_arrs = [np.zeros(self.SHAPE, dtype="float32") for _ in range(2)]
        self.region_arrs[0][:, 0:4, 0:4] = 0.5
        self.region_arrs[1][:, 6:10, 0:4] = 0.5
        query_arr = np.zeros(self.SHAPE, dtype="uint8")
        query_arr[0:3, :, 0:2] = 1  # component 1
        query_arr[6:9, :, 0:2] = 1  # component 2
        self.queryvolume.fetch.return_value = Nifti1Image(query_arr, np.eye(4))
        df = getattr(self, factory)().assign(self.queryvolume)
        self.assertEqual(df[["input structure", "volume"]].values.tolist(), [[1, 0], [1, 1], [2, 0], [2, 1]])

    @parameterized.expand([("statistical_map", ), ("sparse_map", )])
    def test_point_rows_are_ordered_by_point_then_map(self, factory):
        # two regional maps, each containing both points
        self.region_arrs = [np.zeros(self.SHAPE, dtype="float32") for _ in range(2)]
        self.region_arrs[0][0:6] = 0.7
        self.region_arrs[1][3:10] = 0.4
        mp = getattr(self, factory)()
        points = PointCloud([(4, 1, 1), (5, 2, 2)], space=None, sigma_mm=0)
        with patch.object(PointCloud, "space", new_callable=PropertyMock, return_value=self.space), \
                patch.object(PointCloud, "warp", lambda pts, *args, **kwargs: pts), \
                patch.object(Map, "affine", new_callable=PropertyMock, return_value=np.eye(4)):
            df = mp.assign(points)
        self.assertEqual(df[["input structure", "volume"]].values.tolist(), [[0, 0], [0, 1], [1, 0], [1, 1]])

    @parameterized.expand([("statistical_map", ), ("sparse_map", )])
    def test_rejects_query_in_other_space(self, factory):
        mp = getattr(self, factory)()
        self.queryvolume.space = MagicMock()
        with self.assertRaises(ValueError):
            mp.assign(self.queryvolume)
        self.queryvolume.fetch.assert_not_called()


class TestMapAssignVolume(unittest.TestCase):
    """Behaviour of `Map._assign_volume` beyond thresholds and score columns."""

    SHAPE = (10, 10, 10)

    def setUp(self) -> None:
        self.space = MagicMock()
        patcher = patch.object(Map, "space", new_callable=PropertyMock, return_value=self.space)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.region_arr = np.zeros(self.SHAPE, dtype="float32")
        self.region_arr[0:4, 0:4, 0:4] = 0.5
        self.mp = TestMap.get_instance(
            indices={"foo": [{"volume": 0}]},
            volumes=[DummyCls(formats={"nii"}, provides_image=True)],
        )

    def patch_fetch(self, affine: np.ndarray = np.eye(4)) -> None:
        image = Nifti1Image(self.region_arr, affine)
        patcher = patch.object(Map, "fetch", return_value=image)
        patcher.start()
        self.addCleanup(patcher.stop)

    def queryvolume(self, query_arr: np.ndarray, affine: np.ndarray = np.eye(4)) -> MagicMock:
        queryvolume = MagicMock(spec=Volume)
        queryvolume.space = self.space
        queryvolume.fetch.return_value = Nifti1Image(query_arr, affine)
        return queryvolume

    @parameterized.expand([(0, [8, 1]), (1, [8]), (8, [])])
    def test_minsize_voxel_skips_small_components(self, minsize_voxel, expected_sizes):
        self.patch_fetch()
        query_arr = np.zeros(self.SHAPE, dtype="uint8")
        query_arr[0:2, 0:2, 0:2] = 1  # component of 8 voxels
        query_arr[3, 0, 3] = 1  # component of 1 voxel
        assignments = self.mp._assign(self.queryvolume(query_arr), minsize_voxel=minsize_voxel)
        # both components lie within the 64-voxel regional map: size = IoU * 64
        sizes = sorted(
            (int(round(a.intersection_over_union * 64)) for a in assignments), reverse=True
        )
        self.assertEqual(sizes, expected_sizes)

    def test_centroid_is_in_physical_coordinates(self):
        affine = np.array([
            [2., 0., 0., 10.],
            [0., 2., 0., 20.],
            [0., 0., 2., 30.],
            [0., 0., 0., 1.],
        ])
        self.patch_fetch(affine=affine)
        query_arr = np.zeros(self.SHAPE, dtype="uint8")
        query_arr[0:2, 0:2, 0:2] = 1  # voxel centroid (0.5, 0.5, 0.5)
        assignments = self.mp._assign(self.queryvolume(query_arr, affine=affine))
        self.assertEqual(len(assignments), 1)
        np.testing.assert_allclose(assignments[0].centroid, (11., 21., 31.))

    @parameterized.expand([("labelled", True), ("statistical", False)])
    def test_correlation_only_for_statistical_maps(self, _, labelled):
        index_spec = {"volume": 0, "label": 1} if labelled else {"volume": 0}
        mp = TestMap.get_instance(
            indices={"foo": [index_spec]},
            volumes=[DummyCls(formats={"nii"}, provides_image=True)],
        )
        if labelled:
            self.region_arr = (self.region_arr > 0).astype("uint8")
        self.patch_fetch()
        query_arr = np.zeros(self.SHAPE, dtype="uint8")
        query_arr[0:6, 0:2, 0:2] = 1
        with patch.object(Map, "get_region", return_value="foo"):
            df = mp.assign(self.queryvolume(query_arr))
        self.assertEqual(len(df), 1)
        self.assertIn("intersection over union", df.columns)
        self.assertEqual("correlation" in df.columns, not labelled)
