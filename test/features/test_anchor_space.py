from unittest.mock import MagicMock, patch

import pytest

from siibra.commons import Species
from siibra.configuration.factory import Factory
from siibra.core.region import Region
from siibra.core.space import Space
from siibra.features import anchor as _anchor
from siibra.features.anchor import AnatomicalAnchor

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
    location = MagicMock()
    anchor = AnatomicalAnchor(SPECIES, location=location, space=bigbrain)
    assert anchor.matches(bigbrain)
    location.space.matches.assert_not_called()


def test_without_own_space_the_location_space_is_used():
    bigbrain = make_space("BigBrain")
    location = MagicMock()
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
