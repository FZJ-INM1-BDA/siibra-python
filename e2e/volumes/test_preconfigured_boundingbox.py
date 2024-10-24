import siibra
from siibra.features.image.image import Image
from siibra.volumes.volume import Volume
import pytest
from itertools import repeat

map_vols = [v for m in siibra.maps for v in m.volumes]
imagefeatures = [
    feat
    for ftype in siibra.features.Feature._SUBCLASSES[Image]
    for feat in ftype._get_instances()
]
volumes = list(zip(map_vols, repeat(True))) + list(zip(imagefeatures, repeat(False)))


@pytest.mark.parametrize("volume, clip_flag", volumes)
def test_onthefly_and_preconfig_bboxes(volume: Volume, clip_flag: bool):
    configured_bbox = volume._boundingbox
    if configured_bbox is None:
        pytest.skip(f"No preconfigured BoundingBox for {volume} is found.")
    volume._boundingbox = None
    kwargs = {"clip": clip_flag}
    if "neuroglancer/precomputed" in volume.providers:
        kwargs.update(
            {
                "resolution_mm": -1,
                "format": "neuroglancer/precomputed",
                "max_bytes": 2 * 1024**3,
            }
        )
    bbox = volume.get_boundingbox(**kwargs)
    assert configured_bbox == bbox, f" {volume}"
