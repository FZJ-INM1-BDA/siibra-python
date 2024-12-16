import pytest
from itertools import repeat
from time import time
import os
import numpy as np

import siibra
from siibra.features.image.image import Image
from siibra.volumes.volume import Volume

TEST_ALL_PRECONF_BBOXES = eval(os.getenv("TEST_ALL_PRECONF_BBOXES", "False"))
RANDOM_SEED = os.getenv("RANDOM_SEED", None if TEST_ALL_PRECONF_BBOXES else int(time()))
RANDOM_TEST_COUNT = os.getenv("RANDOM_TEST_COUNT", None if TEST_ALL_PRECONF_BBOXES else 20)


map_vols = [v for m in siibra.maps for v in m.volumes]
imagefeatures = [
    feat
    for ftype in siibra.features.Feature._SUBCLASSES[Image]
    for feat in ftype._get_instances()
]
volumes = list(zip(map_vols, repeat(True))) + list(zip(imagefeatures, repeat(False)))

if not TEST_ALL_PRECONF_BBOXES:
    np.random.seed(int(RANDOM_SEED))
    randomly_selected_volumes = [
        volumes[i] for i in np.random.randint(0, len(volumes), int(RANDOM_TEST_COUNT))
    ]


@pytest.mark.parametrize(
    "volume, clip_flag",
    volumes if TEST_ALL_PRECONF_BBOXES else randomly_selected_volumes
)
def test_onthefly_and_preconfig_bboxes(volume: Volume, clip_flag: bool):
    configured_bbox = volume._boundingbox
    if configured_bbox is None:
        pytest.skip(f"No preconfigured BoundingBox for {volume} is found. ")
    volume._boundingbox = None
    kwargs = {"clip": clip_flag}
    if "neuroglancer/precomputed" in volume.providers:
        pytest.xfail("Bounding boxes of 'neuroglancer/precomputed' volumes have to be recalculated due to 0.5 voxel shift fix.")
    bbox = volume.get_boundingbox(**kwargs)
    assert configured_bbox == bbox, f" {volume}"
