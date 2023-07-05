import pytest
import siibra

@pytest.mark.parametrize('space', siibra.spaces)
def test_space(space: siibra.core.space.Space):
    for vol in space.volumes:
        print(vol.providers)

all_spaces = [
    s for s in siibra.spaces
]

@pytest.mark.parametrize('space', all_spaces)
def test_has_desc(space: siibra.core.space.Space):
    assert space.description, f"{space.name!r} does not have desc"

@pytest.mark.parametrize('space', all_spaces)
def test_has_publications(space: siibra.core.space.Space):
    assert len(space.publications) > 0, f"{space.name!r} does not have publication"
