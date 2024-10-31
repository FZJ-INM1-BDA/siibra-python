import pytest
import siibra

from siibra.volumes import Map

preconfiugres_maps = list(siibra.maps)


@pytest.mark.parametrize("mp", preconfiugres_maps)
def test_compute_centroids(mp: Map):
    if not mp.provides_image:
        pytest.skip(
            f"Currently, centroid computation for meshes is not implemented. Skipping '{mp.name}'"
        )
    try:
        _ = mp.compute_centroids()
    except Exception as e:
        pytest.fail(f"Cannot compute all centroids of the map '{mp.name}:'\n{e}")
