import pytest
import siibra


vois = [
    siibra.locations.BoundingBox(
        (-54.90, -29.47, 12.66), (12.69, 35.12, 20.24), "bigbrain"
    ),
    siibra.locations.BoundingBox(
        (-154.90, -29.47, 12.66), (112.69, 38.12, 80.24), "bigbrain"
    ),
    siibra.locations.BoundingBox(
        (-54.90, -29.47, 12.66), (12.69, 38.12, 80.24), "bigbrain"
    ),
]


@pytest.mark.parametrize("voi", vois)
def test_fetching_voi(voi):
    voi.space.get_template().fetch(voi=voi, resolution_mm=0.52)
