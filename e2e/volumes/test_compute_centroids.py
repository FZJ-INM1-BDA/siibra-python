import pytest
import siibra

from siibra.volumes import Map
from siibra.exceptions import NoneCoordinateSuppliedError

preconfiugres_maps = list(siibra.maps)
BLACKLIST = {
    siibra.get_map('julich 1.18', 'mni15', 'statistical'): "SKIP: fails since there are volumes that cannot be fetched."
}


@pytest.mark.parametrize("mp", preconfiugres_maps)
def test_compute_centroids(mp: Map):
    if mp in BLACKLIST.keys():
        pytest.skip(
            f"{BLACKLIST[mp]} Map: {mp.name}"
        )
    if not mp.provides_image:
        pytest.skip(
            f"Currently, centroid computation for meshes is not implemented. Skipping '{mp.name}'"
        )
    try:
        _ = mp.compute_centroids()
    except NoneCoordinateSuppliedError:
        pytest.fail(f"None or NaN centroid calculated for the map '{mp.name}.")
    except Exception as e:
        pytest.fail(f"Cannot compute all centroids of the map '{mp.name} because:'\n{e}")
