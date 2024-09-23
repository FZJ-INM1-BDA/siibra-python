import unittest
import pytest
from unittest.mock import patch, PropertyMock, MagicMock, call
from collections import namedtuple
from itertools import product

from siibra.core.region import Region, RegionRelationAssessments


class TestRegion(unittest.TestCase):
    @staticmethod
    def get_instance(name="foo-bar", children=[]):
        return Region(name, children=children)

    @classmethod
    def setUpClass(cls) -> None:
        cls.child_region = TestRegion.get_instance(name="Area hOc1 (V1, 17, CalcS)")
        cls.parent_region = TestRegion.get_instance(
            name="occipital cortex", children=[cls.child_region]
        )

    def test_regions_init(self):
        self.assertEqual(self.child_region, "Area hOc1 (V1, 17, CalcS)")
        self.assertEqual(self.parent_region, "occipital cortex")

    def test_has_parent(self):
        self.assertFalse(self.parent_region.has_parent(self.child_region))
        self.assertTrue(self.child_region.has_parent(self.parent_region))

    def test_includes(self):
        self.assertTrue(self.parent_region.includes(self.child_region))

    def test_includes_region_false(self):
        self.parent_region.children = []
        self.assertFalse(self.parent_region.includes(self.child_region))

    def test_includes_region_self(self):
        self.assertTrue(self.parent_region.includes(self.parent_region))

    def test_find_child_region(self):
        regions = self.parent_region.find(self.child_region.name)
        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 1)
        self.assertEqual(next(iter(regions)), self.child_region)

    def test_find_child_no_result(self):
        regions = self.child_region.find(self.parent_region.name)
        self.assertIsNotNone(regions)
        self.assertEqual(len(regions), 0)

    def test_matches(self):
        self.assertTrue(self.child_region.matches(self.child_region))
        self.assertTrue(self.child_region.matches("Area hOc1 (V1, 17, CalcS)"))
        self.assertTrue(self.child_region.matches("hoc1"))

        self.assertFalse(self.child_region.matches(self.parent_region))
        self.assertFalse(self.child_region.matches("Area 51"))

    def test_copy(self):
        new_region = Region.copy(self.child_region)
        self.assertFalse(new_region is self.child_region)
        self.assertEqual(new_region.parent, None)


@pytest.fixture
def regions_same_id():
    region = TestRegion.get_instance("foo-bar")
    region1 = TestRegion.get_instance("foo-bar")
    with patch(
        "siibra.core.region.Region.id", new_callable=PropertyMock
    ) as mock_id_prop:
        with patch(
            "siibra.core.region.Region.key", new_callable=PropertyMock
        ) as mock_key_prop:
            mock_id_prop.return_value = "baz"
            mock_key_prop.return_value = "baz.key"
            yield [region, region1]


@pytest.fixture
def regions_different_id():
    regions = [
        TestRegion.get_instance("foo-bar"),
        TestRegion.get_instance("foo-bar"),
    ]

    with patch(
        "siibra.core.region.Region.id", new_callable=PropertyMock
    ) as mock_id_prop:
        with patch(
            "siibra.core.region.Region.key", new_callable=PropertyMock
        ) as mock_key_prop:
            mock_id_prop.side_effect = [
                "foo",
                "bar",
            ]
            mock_key_prop.side_effect = [
                "foo.key",
                "bar.key",
            ]
            yield regions


def test_same_id_mock(regions_same_id):
    for region in regions_same_id:
        assert region.id == "baz"
        assert region.key == "baz.key"


def test_diff_id_mock(regions_different_id):
    id_set = set()
    for region in regions_different_id:
        r_id = region.id
        assert f"{r_id}.key" == region.key
        id_set.add(r_id)
    assert len(id_set) == 2


@pytest.mark.parametrize(
    "regions_same_id,compare_to,expected",
    [
        ["region-standin", "baz", True],
        ["region-standin", "baz.key", True],
        ["region-standin", "foo-bar", True],
        ["region-standin", "hello world", False],
        ["region-standin", None, False],
        ["region-standin", [], False],
        ["region-standin", set(), False],
        ["region-standin", {}, False],
    ],
    indirect=["regions_same_id"],
)
def test_eq_str(regions_same_id, compare_to, expected):
    for region in regions_same_id:
        actual = region == compare_to
        assert actual == expected


def test_eq_region_same_id(regions_same_id):
    assert regions_same_id[0] == regions_same_id[1]


def test_eq_region_diff_id(regions_different_id):
    assert regions_different_id[0] != regions_different_id[1]


fields = (
    "name", "children", "parent", "shortname", "description", "modality",
    "publications", "datasets", "rgb", "spec"
)
defaults = ("", [], None, "", "", "", [], [], None, None)
RegionSpec = namedtuple("RegionSpec", fields, defaults=defaults)


@pytest.mark.parametrize("region_spec", [
    RegionSpec(),
    RegionSpec(spec={"foo": "bar"}),
    RegionSpec(spec={"ebrains": {}}),
    RegionSpec(spec={"ebrains": {"foo": "bar"}}),
])
def test_get_related_regions(region_spec):
    return_val = ["foo"]
    with patch.object(
        RegionRelationAssessments,
        "parse_from_region",
        return_value=return_val
    ) as parse_from_region_mock:
        reg = Region(*region_spec)

        val = [f for f in reg.get_related_regions()]
        parse_from_region_mock.assert_called_once_with(reg)
        assert val == return_val


@pytest.mark.parametrize("spec, test_against, expected_result", [
    (RegionSpec(), "bar", False),
    (RegionSpec(spec={"ebrains": {"foo": "bar"}}), "bar", True),
    (RegionSpec(spec={"ebrains": {"foo": "bar"}}), "foo", False),
])
def test_region_dunder_eq(spec, test_against, expected_result):
    reg = Region(*spec)
    assert (reg == test_against) == expected_result


@pytest.mark.parametrize("id_param, expected_val, raise_flag", [
    ("1", ["1"], False),
    (["1"], ["1"], False),
    (1, None, True),
    ([1], None, True),
    (["1", 1], None, True),
    ({"foo": "bar"}, None, True),
])
def test_parse_id_arg(id_param, expected_val, raise_flag):
    if raise_flag:
        with pytest.raises(Exception):
            RegionRelationAssessments.parse_id_arg(id_param)
        return
    assert RegionRelationAssessments.parse_id_arg(id_param) == expected_val


@pytest.mark.parametrize("long_id, expected_val, error_flag", [
    ("hello world/aabbcc-0099", "aabbcc-0099", False),
    ("aabbcc-0099", "aabbcc-0099", False),
    ({"id": "aabbcc-0099"}, "aabbcc-0099", False),

    ({"foo": "aabbcc-0099"}, None, True),
    ("hello worl", None, True),  // .*codespell-ignore
    ({"id": "hello worlfoo"}, None, True),
])
def test_get_uuid(long_id, expected_val, error_flag):
    if error_flag:
        with pytest.raises(Exception):
            RegionRelationAssessments.get_uuid(long_id)
        return
    assert RegionRelationAssessments.get_uuid(long_id) == expected_val


@pytest.mark.parametrize("obj_path, get_content_value, expected_return_val, error_flag", [
    ("foo-bar", b'{"foo": "bar"}', {"foo": "bar"}, False),
    ("buz/foo-bar", b'{}', {}, False),
    ("foo-gzz", b'[{"foo": "bar"}]', [{"foo": "bar"}], False),
    ("foo-gzz", '[{"foo": "bar"}]', [{"foo": "bar"}], False),
    ("foo-gzz", b'{"ba', None, True),
])
def test_get_obj(obj_path, get_content_value, expected_return_val, error_flag):

    bucket_magic_mock = MagicMock()
    bucket_magic_mock.get_file.return_value.get_content.return_value = get_content_value

    with patch.object(
        RegionRelationAssessments.anony_client.buckets,
        "get_bucket",
        return_value=bucket_magic_mock
    ) as get_bucket_mock:
        if error_flag:
            with pytest.raises(Exception):
                val = RegionRelationAssessments.get_object(obj_path)
            return

        val = RegionRelationAssessments.get_object(obj_path)
        assert val == expected_return_val

        get_bucket_mock.assert_called_once_with("reference-atlas-data")
        bucket_magic_mock.get_file.assert_called_with(obj_path)
        bucket_magic_mock.get_file.return_value.get_content.assert_called_once()


@pytest.mark.parametrize(
    "type_str, _id",
    product(
        ("foo", "bar"), ("1", ["1", "2"])
    )
)
def test_get_snapshot_factory(type_str, _id):
    with patch.object(RegionRelationAssessments, "get_object") as get_obj_mock:
        return_vals = [1, 2]
        get_obj_mock.side_effect = return_vals
        get_objs = RegionRelationAssessments.get_snapshot_factory(type_str)
        _result = [val for val in get_objs(_id)]
        assert _result == return_vals[:len(_id)]
        if isinstance(_id, str):
            get_obj_mock.assert_called_once_with(f"ebrainsquery/v3/{type_str}/{_id}.json")

        if isinstance(_id, list):
            # get_obj_mock.assert_called()
            all_calls = get_obj_mock.call_args_list
            assert len(all_calls) == len(_id)
            assert list(all_calls) == [call(f"ebrainsquery/v3/{type_str}/{i}.json",) for i in _id]
