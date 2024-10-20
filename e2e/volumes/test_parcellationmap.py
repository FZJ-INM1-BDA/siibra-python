import pytest
import siibra

from siibra import MapType
from siibra.volumes import Map
from siibra.volumes.volume import Subvolume

import numpy as np

maps_to_compress = [
    siibra.get_map("2.9", "mni152"),  # contains fragments
    siibra.get_map("difumo 64", "mni152", MapType.STATISTICAL),  # contains subvolumes
]


@pytest.mark.parametrize("siibramap", maps_to_compress)
def test_compress(siibramap: Map):
    assert any(
        [
            any(isinstance(vol, Subvolume) for vol in siibramap.volumes),
            len(siibramap.fragments) > 0,
        ]
    )
    compressed_map = siibramap.compress()
    assert all(
        [
            not any(isinstance(vol, Subvolume) for vol in compressed_map.volumes),
            len(compressed_map.fragments) == 0,
        ]
    )


maps_have_volumes = [
    siibra.get_map("2.9", "mni152", "statistical"),
    siibra.get_map("2.9", "colin", "statistical"),
]


@pytest.mark.parametrize("siibramap", maps_have_volumes)
def test_volume_have_datasets(siibramap: Map):
    all_ds = [ds for volume in siibramap.volumes for ds in volume.datasets]
    assert len(all_ds) > 0


containedness = [
    (
        siibra.Point((27.75, -32.0, 63.725), space="mni152", sigma_mm=3.0),
        siibra.get_map(
            parcellation="julich 2.9", space="mni152", maptype="statistical"
        ),
        "`input containedness` >= 0.5",
    ),
    (
        siibra.Point((27.75, -32.0, 63.725), space="mni152", sigma_mm=0.0),
        siibra.get_map(
            parcellation="julich 2.9", space="mni152", maptype="statistical"
        ),
        "`map value` == None",
    ),
]


@pytest.mark.parametrize("point,map,query", containedness)
def test_containedness(point, map: Map, query):
    assignments = map.assign(point)
    if point.sigma > 0:
        assert len(assignments.query(query)) > 0
    if point.sigma == 0:
        assert len(assignments.query(query)) == 0


def test_point_assignment_to_labelled():
    mp = siibra.get_map('julich 3.0.3', 'mni152')
    # certain point
    point = siibra.Point((25.5, -26.0, 72.0), space='mni152')
    assignments = mp.assign(point)
    assert len(assignments) == 1 and assignments['region'][0].matches("Area 3b (PostCG) right")

    # uncertain point
    point_uncertain = siibra.Point((25.5, -26.0, 72.0), space='mni152', sigma_mm=3.)
    assignments_uncertain = mp.assign(point_uncertain).sort_values(by=['input containedness'], ascending=False)
    assignments_uncertain['region'][0].matches("Area 4a (PreCG) right")


maps_w_fragments = (
    (siibra.get_map("julich 2.9", "mni152", "labelled"), "nii"),
    (siibra.get_map("julich 2.9", "colin27", "labelled"), "nii"),
    (siibra.get_map("julich 2.9", "colin27", "labelled"), "neuroglancer/precomputed"),
)


@pytest.mark.parametrize("siibramap, format", maps_w_fragments)
def test_merged_fragment_shape(siibramap: Map, format):
    # NOTE: This test is very specific to Julich Brain 2.9. Not all fragments will have the same shape...
    vol_l = siibramap.fetch(fragment="left hemisphere", format=format)
    vol_r = siibramap.fetch(fragment="right hemisphere", format=format)
    vol_b = siibramap.fetch(format=format)  # auto-merged map
    assert vol_l.dataobj.dtype == vol_r.dataobj.dtype == vol_b.dataobj.dtype, f"Map: {siibramap}"
    assert vol_l.dataobj.shape == vol_r.dataobj.shape == vol_b.dataobj.shape, f"Map: {siibramap}"


def test_region_1to1ness_in_parcellation():
    failed = []
    for m in Map.registry():
        parc = m.parcellation
        for region_name in m.regions:
            try:
                _ = parc.get_region(region_name)
            except Exception:
                failed.append(
                    {
                        "region name": region_name,
                        "map": m.name
                    }
                )
    assert len(failed) == 0, print("The regions in maps that can't be gotten in respective parcellations\n", failed)


def test_fetching_merged_volume():
    # this also tests for volume.merge as it is used in Map.fetch
    mp = siibra.get_map("julich 2.9", "bigbrain")
    assert len(mp) > 1
    _ = mp.fetch()


# TODO: add tests for jba 3.1
@pytest.mark.parametrize("siibramap", [
    siibra.get_map('julich 3.0.3', 'fsaverage'),
    siibra.get_map('julich 3.0.3', 'mni152'),
])
def test_fetching_mask(siibramap: Map):
    for fmt in siibramap.formats:
        for region in siibramap.regions:
            mesh_or_img = siibramap.fetch(region, format=fmt)
            arr = mesh_or_img['labels'] if isinstance(mesh_or_img, dict) else mesh_or_img.dataobj
            unq_vals = np.unique(arr)
            assert np.array_equal(unq_vals, [0, 1]), (
                f"{siibramap}, {fmt}. Mask for {region} should only contain 0 "
                f"and 1 but found: {unq_vals}"
            )


def test_freesurfer_annot_map_fetch():
    annotmaps = [
        mp for mp in siibra.maps
        if any(
            fmt in ["freesurfer-annot", "zip/freesurfer-annot"]
            for fmt in mp.formats
        )
    ]
    if len(annotmaps) == 0:
        pytest.skip("No freesurfer-annot map in the configuration.")
    for mp in annotmaps:
        mesh = mp.fetch(fragment='left', variant='pial')
    assert np.array_equal(
        np.unique(mesh['labels']),
        np.array([0] + list(mp.labels))
    )
