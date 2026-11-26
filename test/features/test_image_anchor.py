from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from siibra.commons import Species
from siibra.core.assignment import Qualification
from siibra.core.region import Region
from siibra.core.space import Space
from siibra.features.anchor import AnatomicalAnchor
from siibra.features.image.image import Image
from siibra.volumes.volume import Volume


def make_space(name):
    space = MagicMock(spec=Space)
    space.name = name
    space.species = Species.decode("Homo sapiens")
    space.matches.side_effect = lambda other: other is space
    return space


@pytest.fixture
def bigbrain():
    """Images in a BigBrain stand-in, with a bounding box that counts its computations."""
    space = make_space("BigBrain")
    box = MagicMock()
    box.space = space
    with patch.object(Volume, "space", new_callable=PropertyMock, return_value=space), \
            patch.object(Volume, "get_boundingbox", return_value=box) as get_boundingbox:
        yield space, box, get_boundingbox


def make_image(region=None):
    return Image(name="VOI", modality="cell body staining", space_spec={"@id": "bigbrain"}, providers=[], region=region)


def test_image_anchor_has_the_image_space_and_species(bigbrain):
    space, _, get_boundingbox = bigbrain
    image = make_image()
    assert isinstance(image.anchor, AnatomicalAnchor)
    assert image.anchor.space is space
    assert image.anchor.species == {Species.decode("Homo sapiens")}
    get_boundingbox.assert_not_called()


def test_space_queries_do_not_compute_the_bounding_box(bigbrain):
    space, _, get_boundingbox = bigbrain
    image = make_image()
    assert image.matches(space)
    assert not image.matches(make_space("MNI 152"))
    assert str(image.anchor) == "in BigBrain"
    get_boundingbox.assert_not_called()


def test_the_unclipped_bounding_box_is_the_location(bigbrain):
    _, box, get_boundingbox = bigbrain
    image = make_image()
    assert image.anchor.location is box
    assert image.anchor.location is box
    get_boundingbox.assert_called_once_with(clip=False)


def test_region_queries_use_the_region_of_an_image_if_given(bigbrain):
    _, _, get_boundingbox = bigbrain
    image = make_image(region="hOc1 left")
    image.anchor._regions_cached = {Region("hOc1 left"): Qualification.EXACT}  # instead of looking it up
    with patch.object(Region, "assign", return_value=None):
        assert not image.matches(Region("hOc2 left"))
    get_boundingbox.assert_not_called()


def test_region_queries_use_the_bounding_box_of_an_image_without_region(bigbrain):
    _, box, get_boundingbox = bigbrain
    box.assign.return_value = "box overlaps region"
    query = Region("hOc1 left")
    assert make_image().anchor.assign(query) == ["box overlaps region"]
    box.assign.assert_called_once_with(query)


def test_description_names_the_anchor_without_the_bounding_box(bigbrain):
    _, _, get_boundingbox = bigbrain
    image = make_image()
    assert image.description == "Image feature with modality cell body staining, in BigBrain"
    get_boundingbox.assert_not_called()
