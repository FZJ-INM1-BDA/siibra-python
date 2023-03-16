import pytest
import siibra

@pytest.mark.parametrize('space', siibra.spaces)
def test_space(space: siibra.core.space.Space):
    for vol in space.volumes:
        print(vol.providers)

should_have_desc = [
    siibra.spaces['big brain']
]

@pytest.mark.parametrize('space', should_have_desc)
def test_has_desc(space: siibra.core.space.Space):
    assert space.description
