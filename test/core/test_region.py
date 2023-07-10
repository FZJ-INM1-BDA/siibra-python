import unittest
import pytest
from siibra.core.region import Region
from unittest.mock import patch, PropertyMock


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


# TODO move these into int tests

# all_regions = [
#     pytest.param(r, marks=pytest.mark.xfail(reason="Parent is not Region, xfail for now")) if not isinstance(r.parent, Region) else
#     r
#     for parc in parcellations
#     for r in parc]

# @pytest.mark.parametrize('region', all_regions)
# def test_region_to_model(region: Region):
#     model = region.to_model()

#     # TODO some region have space in their id...
#     # Please sanitize then uncomment this test
#     # import re
#     # assert re.match(r"^[\w/\-.:]+$", model.id), f"model_id should only contain [\w/\-.:]+, but is instead {model.id}"

# detailed_region=[
#     ("julich 2.9", "hoc1 left", "mni152", False, True),
#     ("julich 2.9", "hoc1 right", "mni152", False, True),
#     ("julich 2.9", "hoc1 left", "colin 27", False, True),
#     ("julich 2.9", "hoc1 right", "colin 27", False, True),
#     ("julich 2.9", "hoc1 right", "fsaverage", False, False),
#     pytest.param(
#         "julich 2.9", "hoc1", "bigbrain", False, True,
#         marks=pytest.mark.xfail(reason="big brain returning 2 centoids... what?"),
#     ),
#     ("julich 2.9", "hoc1 right", "bigbrain", True, None),
# ]

# @pytest.mark.parametrize('parc_spec,region_spec,space_spec,expect_raise,expect_best_view_point', detailed_region)
# def test_detail_region(parc_spec,region_spec,space_spec,expect_raise,expect_best_view_point):
#     p = siibra.parcellations[parc_spec]
#     r = p.get_region(region_spec)
#     s = siibra.spaces[space_spec]
#     if expect_raise:
#         with pytest.raises(RuntimeError):
#             model = r.to_model(detail=True, space=s)
#         return

#     model = r.to_model(detail=True, space=s)
#     assert model.has_parent is not None
#     assert model.has_annotation is not None
#     assert (model.has_annotation.best_view_point is not None) == expect_best_view_point
#     assert model.version_innovation is not None

# has_inspired_by = [
#     ("julich 2.9", "hoc1 left", "mni152"),
#     ("long bundle", "Left short cingulate fibres", "mni152")
# ]

# @pytest.mark.parametrize('parc_spec, region_spec, space_spec', has_inspired_by)
# def test_has_inspired_by(parc_spec, region_spec, space_spec):
#     p = siibra.parcellations[parc_spec]
#     r = p.get_region(region_spec)
#     model = r.to_model(space=siibra.spaces[space_spec])
#     assert model.has_annotation.visualized_in is not None, f"expecting has_annotation.visualized_in is defined"

#     # has_annotation.visualized_in is either in region.volumes or parcellation.volumes
#     assert any(vol.model_id == model.has_annotation.visualized_in["@id"] for vol in [*r.volumes, *p.volumes])

# has_internal_identifier = [
#     ('julich 2.9', "hoc1", "big brain", True),
#     ('julich 2.9', "hoc1", "mni152", False),
#     ('julich 2.9', "hoc1 left", "mni152", True),
#     ('julich 2.9', "hoc1 left", "fsaverage", True),

#     ('julich 2.9', "fp1", "big brain", False),
#     ('julich 2.9', "fp1", "mni152", False),
#     ('julich 2.9', "fp1 left", "mni152", True),

#     ('difumo 64', "cuneus", "mni152", True),
#     ('Short Fiber Bundles - HCP', "rh_Tr-Tr_1", "mni152", True),
#     # ('isocortex', 'Isocortex', 'big brain', True),
# ]
# @pytest.mark.parametrize('parc_spec, region_spec, space_spec, expect_id_defined', has_internal_identifier)
# def test_has_internal_identifier(parc_spec, region_spec, space_spec, expect_id_defined):
#     p = siibra.parcellations[parc_spec]
#     r = p.get_region(region_spec)
#     model = r.to_model(space=siibra.spaces[space_spec])
#     assert model.has_annotation.internal_identifier is not None, f"expecting has_annotation.internal_identifier is defined"
#     assert (model.has_annotation.internal_identifier != "unknown") == expect_id_defined, f"expect_id_defined: {expect_id_defined}, but id: {model.has_annotation.internal_identifier}"
#     # assert (model.has_annotation.visualized_in is not None) == expect_id_defined

# jba29_bigbrain = [
#     ("hoc1", 1),
#     ("hoc2", 1),
#     ("hoc5", 1),
#     ("hIP7", 18),
#     ("MGB-MGBd (CGM, Metathalamus)", 1),
# ]
# big_brain = siibra.spaces['big brain']
# get_labelidx = re.compile(r'\#([1-9]+)$')

# @pytest.mark.parametrize("region_spec,expected_labelindex", jba29_bigbrain)
# def test_bigbrain_jba29_has_labelindex(region_spec: str, expected_labelindex: int):
#     p = siibra.parcellations['2.9']
#     r: Region = p.find_regions(region_spec)[0]
#     model = r.to_model(space=big_brain)

#     assert model.has_annotation.visualized_in is not None, f"expect visualized_in to be defined"

#     found = [insp
#         for insp in model.has_annotation.inspired_by
#         if "siibra_python_ng_precomputed_labelindex://" in insp.get("@id")]
#     assert len(found) == 1, f"expecting one and only 1 labelindex metadata"
#     label = get_labelidx.search(found[0].get("@id"))
#     assert label is not None, f"regex should match"
#     assert label[1] == str(expected_labelindex), f"expected label index: {expected_labelindex}, actual: {get_labelidx[1]}"

# if __name__ == "__main__":
#     unittest.main()
