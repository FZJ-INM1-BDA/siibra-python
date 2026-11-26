from unittest.mock import MagicMock, NonCallableMagicMock, patch

import pytest

from siibra.commons import Species
from siibra.configuration.factory import Factory
from siibra.core.assignment import Qualification
from siibra.core.region import Region
from siibra.core.space import Space
from siibra.features import anchor as _anchor
from siibra.features.anchor import AnatomicalAnchor
from siibra.locations.location import Location

SPECIES = Species.UNSPECIFIED_SPECIES


def make_space(name):
    space = MagicMock(spec=Space)
    space.name = name
    space.matches.side_effect = lambda other: other is space
    return space


def test_space_query_matches_the_anchor_space():
    bigbrain, mni = make_space("BigBrain"), make_space("MNI 152")
    anchor = AnatomicalAnchor(SPECIES, space=bigbrain)
    assert anchor.space is bigbrain
    assert anchor.matches(bigbrain)
    assert not anchor.matches(mni)


def test_space_alone_does_not_match_regions():
    anchor = AnatomicalAnchor(SPECIES, space=make_space("BigBrain"))
    assert not anchor.matches(Region("hOc1 left"))


def test_own_space_decides_without_the_location():
    bigbrain = make_space("BigBrain")
    location = NonCallableMagicMock(spec=Location)
    anchor = AnatomicalAnchor(SPECIES, location=location, space=bigbrain)
    assert anchor.matches(bigbrain)
    location.space.matches.assert_not_called()


def test_without_own_space_the_location_space_is_used():
    bigbrain = make_space("BigBrain")
    location = NonCallableMagicMock(spec=Location)
    location.space = bigbrain
    anchor = AnatomicalAnchor(SPECIES, location=location)
    assert anchor.space is bigbrain
    assert anchor.matches(bigbrain)


def test_space_spec_is_resolved_from_the_registry():
    bigbrain = make_space("BigBrain")
    with patch.object(_anchor, "get_registry") as get_registry:
        get_registry.return_value.get.return_value = bigbrain
        anchor = AnatomicalAnchor(SPECIES, space="bigbrain-id")
        assert anchor.space is bigbrain
        assert anchor.space is bigbrain
    get_registry.assert_called_once_with("Space")
    get_registry.return_value.get.assert_called_once_with("bigbrain-id")


def test_space_only_anchor_is_described_by_its_space():
    assert str(AnatomicalAnchor(SPECIES, space=make_space("BigBrain"))) == "in BigBrain"


def test_adding_anchors_keeps_a_shared_space():
    bigbrain = make_space("BigBrain")
    combined = AnatomicalAnchor(SPECIES, space=bigbrain) + AnatomicalAnchor(SPECIES, space=bigbrain)
    assert combined.space is bigbrain
    different = AnatomicalAnchor(SPECIES, space=bigbrain) + AnatomicalAnchor(SPECIES, space=make_space("MNI 152"))
    assert different.space is None


def test_extract_anchor_with_only_a_space():
    anchor = Factory.extract_anchor({"species": "Mus musculus"}, space="allen-ccf-id")
    assert anchor._space_spec == "allen-ccf-id"
    with pytest.raises(RuntimeError, match="neither region, location nor space"):
        Factory.extract_anchor({"species": "Mus musculus"})


def test_a_deferred_location_is_computed_once_when_first_needed():
    bigbrain = make_space("BigBrain")
    box = MagicMock()
    box.space = bigbrain
    compute = MagicMock(return_value=box)
    anchor = AnatomicalAnchor(SPECIES, location=compute)
    compute.assert_not_called()
    assert anchor.location is box
    assert anchor.location is box
    compute.assert_called_once_with()


def test_space_queries_do_not_compute_a_deferred_location():
    bigbrain = make_space("BigBrain")
    compute = MagicMock()
    anchor = AnatomicalAnchor(SPECIES, location=compute, space=bigbrain)
    assert anchor.matches(bigbrain)
    assert not anchor.matches(make_space("MNI 152"))
    assert str(anchor) == "in BigBrain"
    compute.assert_not_called()


def test_region_queries_with_a_region_do_not_compute_a_deferred_location():
    compute = MagicMock()
    anchor = AnatomicalAnchor(SPECIES, location=compute, region="hOc1 left", space=make_space("BigBrain"))
    anchor._regions_cached = {Region("hOc1 left"): Qualification.EXACT}  # instead of looking it up
    with patch.object(Region, "assign", return_value=None):
        assert not anchor.matches(Region("hOc2 left"))
    compute.assert_not_called()
    assert str(anchor) == "hOc1 left in BigBrain"


def test_region_queries_without_a_region_use_the_deferred_location():
    query = Region("hOc1 left")
    box = MagicMock()
    box.assign.return_value = "box assignment"
    anchor = AnatomicalAnchor(SPECIES, location=lambda: box, space=make_space("BigBrain"))
    assert anchor.assign(query) == ["box assignment"]
    box.assign.assert_called_once_with(query)


def test_invalid_location():
    with pytest.raises(ValueError, match="Invalid location"):
        AnatomicalAnchor(SPECIES, location="somewhere")
