import pytest
import siibra

from siibra import MapType
from siibra.volumes import Map
from siibra.volumes.volume import Subvolume

maps_to_compress = [
    siibra.get_map("2.9", "mni152"), # contains fragments
    siibra.get_map("difumo 64", "mni152", MapType.STATISTICAL), # contains subvolumes
]

@pytest.mark.parametrize('siibramap', maps_to_compress)
def test_compress(siibramap: Map):
    assert any([
        any(isinstance(vol, Subvolume) for vol in siibramap.volumes),
        len(siibramap.fragments) > 0,
    ])
    compressed_map = siibramap.compress()
    assert all([
        not any(isinstance(vol, Subvolume) for vol in compressed_map.volumes),
        len(compressed_map.fragments) == 0,
    ])

maps_have_volumes = [
    siibra.get_map("2.9", "mni152", "statistical"),
    siibra.get_map("2.9", "colin", "statistical"),
]

@pytest.mark.parametrize('siibramap', maps_have_volumes)
def test_volume_have_datasets(siibramap: Map):
    all_ds = [ds
        for volume in siibramap.volumes
        for ds in volume.datasets]
    assert len(all_ds) > 0

containedness = [
    (
        siibra.Point((27.75, -32.0, 63.725), space='mni152', sigma_mm=3.),
        siibra.get_map(
            parcellation="julich 2.9",
            space="mni152",
            maptype="statistical"
        ),
        '`input containedness` >= 0.5'
    )
]

@pytest.mark.parametrize('point,map,query', containedness)
def test_containedness(point,map:Map,query):
    assignments = map.assign(point)
    assert len(assignments.query(query)) > 0
