import pytest
import os
from time import time

import numpy as np

import siibra
from siibra.volumes import Map
from siibra.exceptions import NoneCoordinateSuppliedError

TEST_ALL_MAPS = eval(os.getenv("TEST_ALL_MAPS", "False"))

preconfiugres_maps = list(siibra.maps)
BLACKLIST = {
    siibra.get_map('julich 1.18', 'mni15', 'statistical'): "SKIP: fails since there are volumes that cannot be fetched."
}

if not TEST_ALL_MAPS:
    RANDOM_SEED = os.getenv("RANDOM_SEED", None if TEST_ALL_MAPS else int(time()))
    RANDOM_TEST_COUNT = os.getenv("RANDOM_TEST_COUNT", None if TEST_ALL_MAPS else 3)
    np.random.seed(int(RANDOM_SEED))
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print(f"RANDOM_TEST_COUNT: {RANDOM_TEST_COUNT}")
    randomly_selected_maps = [
        preconfiugres_maps[i] for i in np.random.randint(0, len(preconfiugres_maps), int(RANDOM_TEST_COUNT))
    ]


@pytest.mark.parametrize(
    "mp",
    preconfiugres_maps if TEST_ALL_MAPS else randomly_selected_maps
)
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
