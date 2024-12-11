import pytest
from typing import Tuple
import siibra


@pytest.mark.parametrize("space", siibra.spaces)
def test_space(space: siibra.core.space.Space):
    for vol in space.volumes:
        print(vol.providers)


should_have_desc = [siibra.spaces["big brain"]]


@pytest.mark.parametrize("space", should_have_desc)
def test_has_desc(space: siibra.core.space.Space):
    assert space.description


voi_params = [
    ("mni152", "nii", (57, 56, 57), {}),
    ("mni152", "neuroglancer/precomputed", (29, 28, 29), {}),
    ("mni152", "neuroglancer/precomputed", (57, 56, 57), {"resolution_mm": 1.0}),
    ("bigbrain", "neuroglancer/precomputed", (42, 45, 42), {}),
]


@pytest.mark.parametrize("space_spec, format, shape, kwargs", voi_params)
def test_voi_fetching(
    space_spec: str, format: str, shape: Tuple[int, int, int], kwargs
):
    space = siibra.spaces.get(space_spec)
    voi = siibra.locations.BoundingBox(
        [-12.033761221226529, -45.77021322122653, -0.28205922122652827],
        [43.50666522122653, 9.770213221226529, 55.25836722122653],
        space,
    )
    template = space.get_template()
    img = template.fetch(format=format, voi=voi, **kwargs)
    assert (
        img.shape == shape
    ), f"Expected fetched voi to have the shape {shape} but has {img.shape}"
