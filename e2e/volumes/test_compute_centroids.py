import pytest
import siibra

from siibra.volumes import Map

preconfiugres_maps = list(siibra.maps)


@pytest.mark.parametrize("mp", preconfiugres_maps)
def test_compute_centroids(mp: Map):
    _ = mp.compute_centroids()
