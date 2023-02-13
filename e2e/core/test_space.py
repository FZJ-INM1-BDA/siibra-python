import pytest
import siibra

@pytest.mark.parametrize('space', siibra.spaces)
def test_space(space: siibra.core.space.Space):
    for vol in space.volumes:
        print(vol.providers)