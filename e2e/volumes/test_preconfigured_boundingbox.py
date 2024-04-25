import siibra
from siibra.features.image.image import Image
from siibra.volumes.volume import Volume
import pytest

space_vols = [v for s in siibra.spaces for v in s.volumes]
map_vols = [v for m in siibra.maps for v in m.volumes]
imagefeatures = [
    feat
    for ftype in siibra.features.Feature._SUBCLASSES[Image]
    for feat in ftype._get_instances()
]
volumes = space_vols + map_vols + imagefeatures


@pytest.mark.parametrize("volume", volumes)
def test_onthefly_and_preconfig_bboxes(volume: Volume):
    configured_bbox = volume._boundingbox
    if configured_bbox is None:
        pytest.skip(f"No preconfigured BoundingBox for {volume} is found.")
    volume._boundingbox = None
    kwargs = {"clip": True}
    if "neuroglancer/precomputed" in volume.providers:
        kwargs.update({
            "clip": False,
            "resolution_mm": -1,
            "format": "neuroglancer/precomputed",
            "max_bytes": 2 * 1024**3
        })
    bbox = volume.get_boundingbox(**kwargs)
    assert configured_bbox == bbox, f'{volume}'
