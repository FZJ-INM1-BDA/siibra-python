"""Unit tests for the Geometry base class, backed by an in-memory provider."""

import logging

import numpy as np
import pytest

from siibra.core.region import Region
from siibra.geometries import Geometry, GeometryProvider


class ArrayProvider(GeometryProvider):
    """Holds one (N, 3) coordinate array per fragment. Not registered: no srctype in the class statement."""

    srctype = "test/array"

    def __init__(self, arrays):
        self.arrays = arrays

    @property
    def fragments(self):
        return list(self.arrays)

    def fetch(self, fragments=None):
        names = list(self.arrays) if fragments is None else fragments
        return {name: self.arrays[name] for name in names}

    def get_coordinates(self, fragments=None):
        return np.concatenate(list(self.fetch(fragments).values()))

    @property
    def _url(self):
        return {name: f"memory://{name}" for name in self.arrays}


HOC1_LEFT, HOC1_RIGHT, HOC2_LEFT = Region("hOc1 left"), Region("hOc1 right"), Region("hOc2 left")
HOC1 = Region("hOc1", children=[HOC1_LEFT, HOC1_RIGHT])
ROOT = Region("root", children=[HOC1, HOC2_LEFT])

ARRAYS = {
    "Left_Arcuate": np.array([[0.0, 0, 0], [1, 1, 1]]),
    "Left_Arcuate_anterior": np.array([[2.0, 2, 2]]),
    "Right_Uncinate": np.array([[-1.0, 5, 3], [4, -2, 0]]),
}
LINKS = {
    "Left_Arcuate": [HOC1_LEFT],
    "Left_Arcuate_anterior": [HOC1_RIGHT, HOC2_LEFT],
    "Right_Uncinate": [HOC2_LEFT],
}


@pytest.fixture
def geometry():
    return Geometry({}, [ArrayProvider(ARRAYS)], name="bundles", fragment_regions=LINKS)


def test_providers_are_listed_by_source_type(geometry):
    assert geometry.providers == {"test/array": ArrayProvider(ARRAYS)._url}


def test_needs_a_provider():
    with pytest.raises(ValueError):
        Geometry({}, [])


def test_rejects_duplicate_source_types():
    with pytest.raises(ValueError):
        Geometry({}, [ArrayProvider(ARRAYS), ArrayProvider(ARRAYS)])


def test_rejects_malformed_affine():
    with pytest.raises(ValueError):
        Geometry({}, [ArrayProvider(ARRAYS)], affine=np.eye(3))


def test_fragments_come_from_the_provider(geometry):
    assert geometry.fragments == list(ARRAYS)


def test_fetch_without_selection_returns_everything(geometry):
    assert list(geometry.fetch()) == list(ARRAYS)


def test_exact_fragment_name_wins_over_substring(geometry):
    assert list(geometry.fetch(fragment="Left_Arcuate")) == ["Left_Arcuate"]


@pytest.mark.parametrize("spec, expected", [
    ("left_arcuate", "Left_Arcuate"),
    ("anterior", "Left_Arcuate_anterior"),
    ("uncinate", "Right_Uncinate"),
])
def test_fragment_matching_is_case_insensitive(geometry, spec, expected):
    assert list(geometry.fetch(fragment=spec)) == [expected]


def test_ambiguous_fragment_raises(geometry):
    with pytest.raises(ValueError, match="ambiguous"):
        geometry.fetch(fragment="Left")


def test_unknown_fragment_raises(geometry):
    with pytest.raises(ValueError, match="no fragment"):
        geometry.fetch(fragment="Corpus_Callosum")


def test_several_fragments_are_deduplicated(geometry):
    selected = geometry.fetch(fragment=["Right_Uncinate", "uncinate", "Left_Arcuate"])
    assert list(selected) == ["Right_Uncinate", "Left_Arcuate"]


def test_fragment_and_region_cannot_be_combined(geometry):
    with pytest.raises(ValueError, match="not both"):
        geometry.fetch(fragment="Left_Arcuate", region=HOC1)


@pytest.mark.parametrize("region, expected", [
    (HOC1_LEFT, ["Left_Arcuate"]),
    (HOC2_LEFT, ["Left_Arcuate_anterior", "Right_Uncinate"]),
    (HOC1, ["Left_Arcuate", "Left_Arcuate_anterior"]),  # collects subregions
    (ROOT, list(ARRAYS)),
])
def test_get_fragments_by_region(geometry, region, expected):
    assert geometry.get_fragments(region) == expected


def test_get_fragments_resolves_region_names(geometry):
    assert geometry.get_fragments("hOc1 left") == ["Left_Arcuate"]
    assert geometry.get_fragments("hOc1") == ["Left_Arcuate", "Left_Arcuate_anterior"]


def test_region_names_resolve_exactly_when_a_parent_also_matches():
    # every child of this root matches "hOc1", so find() returns the root as well
    left, right = Region("hOc1 left"), Region("hOc1 right")
    Region("single area root", children=[Region("hOc1", children=[left, right])])
    geometry = Geometry({}, [ArrayProvider(ARRAYS)], fragment_regions={"Left_Arcuate": [left], "Right_Uncinate": [right]})
    assert geometry.get_fragments("hOc1") == ["Left_Arcuate", "Right_Uncinate"]
    assert geometry.get_fragments("hOc1 left") == ["Left_Arcuate"]


def test_unresolvable_region_name_raises(geometry):
    with pytest.raises(ValueError, match="Cannot resolve"):
        geometry.get_fragments("hOc5")


def test_fetch_by_region(geometry):
    assert list(geometry.fetch(region=HOC2_LEFT)) == ["Left_Arcuate_anterior", "Right_Uncinate"]


def test_fetch_by_region_does_not_select_fragments_of_parent_regions():
    # linked to hOc1 as a whole: nothing says the fragment ends in its left part
    geometry = Geometry({}, [ArrayProvider(ARRAYS)], fragment_regions={"Left_Arcuate": [HOC1]})
    assert geometry.get_fragments(HOC1_LEFT) == []
    with pytest.raises(ValueError, match="No fragment"):
        geometry.fetch(region=HOC1_LEFT)


def test_get_fragments_requires_links():
    with pytest.raises(ValueError, match="not linked"):
        Geometry({}, [ArrayProvider(ARRAYS)]).get_fragments(HOC1)


def test_coordinates_and_boundingbox(geometry):
    coords = geometry.get_coordinates(fragment="Right_Uncinate")
    np.testing.assert_array_equal(coords, ARRAYS["Right_Uncinate"])
    bbox = geometry.get_boundingbox(fragment="Right_Uncinate")
    np.testing.assert_array_equal(bbox.minpoint.coordinate, [-1, -2, 0])
    np.testing.assert_array_equal(bbox.maxpoint.coordinate, [4, 5, 3])


def test_affine_is_applied_to_coordinates():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [10, 0, 0]
    geometry = Geometry({}, [ArrayProvider(ARRAYS)], affine=affine)
    np.testing.assert_array_equal(
        geometry.get_coordinates(fragment="Left_Arcuate"),
        [[10, 0, 0], [12, 2, 2]],
    )
    # the stored data is untouched
    np.testing.assert_array_equal(ARRAYS["Left_Arcuate"], [[0, 0, 0], [1, 1, 1]])


def test_unsupported_format_raises(geometry):
    with pytest.raises(ValueError, match="format"):
        geometry.fetch(format="mesh")


def test_mismatched_fragment_links_are_reported_once(caplog):
    links = {"Left_Arcuate": [HOC1_LEFT], "Corpus_Callosum": [HOC1]}
    geometry = Geometry({}, [ArrayProvider(ARRAYS)], fragment_regions=links)
    with caplog.at_level(logging.WARNING, logger="siibra"):
        geometry.fetch()
        geometry.fetch(region=HOC1_LEFT)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 2
    assert "not in the data" in messages[0] and "Corpus_Callosum" in messages[0]
    assert "not linked to any region" in messages[1] and "Right_Uncinate" in messages[1]
