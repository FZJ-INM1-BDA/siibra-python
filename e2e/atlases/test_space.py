import pytest
import siibra


@pytest.mark.parametrize("space", siibra.spaces)
def test_space(space: siibra.Space):
    assert len(space.volumes) > 0


should_have_desc = ["bigbrain"]

@pytest.mark.parametrize("space_spec", should_have_desc)
def test_has_desc(space_spec: str):
    space = siibra.get_space(space_spec)
    assert space.description
