import pytest
import siibra

from siibra import MapType
from siibra.volumes import Map
from siibra.volumes.sparsemap import SparseMap
from siibra.volumes.volume import Subvolume

from itertools import product

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

# TODO: when merging neuroglancer/precomputed is supported, add to the list
maps_w_fragments = product(
    (
        siibra.get_map("julich 2.9", "mni152", "labelled"),
        siibra.get_map("julich 2.9", "colin", "labelled")
    ),
    ("nii",)
)

@pytest.mark.parametrize('siibramap, format', maps_w_fragments)
def test_merged_fragment_shape(siibramap: Map, format):
    vol_l = siibramap.fetch(fragment="left hemisphere", format=format)
    vol_r = siibramap.fetch(fragment="right hemisphere", format=format)
    vol_b = siibramap.fetch(format=format)  # auto-merged map
    assert vol_l.dataobj.dtype == vol_r.dataobj.dtype == vol_b.dataobj.dtype
    assert vol_l.dataobj.shape == vol_r.dataobj.shape == vol_b.dataobj.shape

def test_sparsemap_cache_uniqueness():
    
    mp157 = siibra.get_map("julich 3.0", "colin 27", "statistical", spec="157")
    mp175 = siibra.get_map("julich 3.0", "colin 27", "statistical", spec="175")
    assert mp157.sparse_index.probs[0] != mp175.sparse_index.probs[0]

# checks labelled/statistical returns volume size matches template
# see https://github.com/FZJ-INM1-BDA/siibra-python/issues/302
MNI152_ID="minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"
COLIN_ID="minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992"

JBA_29_ID="minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290"
JBA_30_ID="minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-300"

HOC1_RIGHT="Area hOc1 (V1, 17, CalcS) - right hemisphere"
FP1_RIGHT="Area Fp1 (FPole) - right hemisphere"

STATISTIC_ENDPOINT="statistical"
LABELLED_ENDPOINT="labelled"

map_shape_args = product(
    ((MNI152_ID, (193, 229, 193)),),
    (JBA_29_ID,),
    (STATISTIC_ENDPOINT, LABELLED_ENDPOINT),
    (HOC1_RIGHT, FP1_RIGHT, None),
)

@pytest.mark.parametrize('space_shape,parc_id,map_endpoint,region_name', map_shape_args)
def test_map_shape(space_shape,parc_id,map_endpoint,region_name):
    if region_name is None and map_endpoint == STATISTIC_ENDPOINT:
        assert True
        return
    space_id, expected_shape = space_shape
    
    volume_data = None
    if region_name is not None:
        region = siibra.get_region(parc_id, region_name)
        volume_data = region.fetch_regional_map(space_id, map_endpoint)
    else:
        labelled_map = siibra.get_map(parc_id, space_id, map_endpoint)
        assert labelled_map is not None
        volume_data = labelled_map.fetch()
    
    assert volume_data
    assert volume_data.get_fdata().shape == expected_shape, f"{volume_data.get_fdata().shape}, {expected_shape}, {region_name}, {map_endpoint}, {space_id}"
