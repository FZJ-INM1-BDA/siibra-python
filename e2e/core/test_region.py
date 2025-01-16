import pytest
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import siibra
from siibra.core.assignment import Qualification
from siibra.core.region import Region


regions = [
    siibra.get_region("julich 3.0", "Area 4p (PreCG) right"),
    siibra.get_region("julich 3.0", "hoc1 left"),
]


@pytest.mark.parametrize("region", regions)
def test_region_spatial_props(region: Region):
    props = region.spatial_props("mni152")
    for idx, cmp in enumerate(props, start=1):
        assert cmp.volume >= props[idx - 1].volume


regions_of_concern = [
    ("julich 3.0", "frontal lobe"),
    ("julich 1.18", "Ch 123 (Basal Forebrain) left"),
    ("waxholm 4", "lateral olfactory tract"),
]


# Test duplicate named regions and regions with only child
@pytest.mark.parametrize("parc_spec, region_name", regions_of_concern)
def test_get_region(parc_spec, region_name):
    region = siibra.get_region(parc_spec, region_name)
    assert region
    assert region.name == region_name


regionspecs = [
    ("julich 3.0", "julich 3.0", 1, [siibra.parcellations["julich 3.0"].name]),
    (
        "waxholm 4",
        "lateral olfactory tract",
        2,
        ["lateral olfactory tract", "Nucleus of the lateral olfactory tract"],
    ),
    (
        "julich 2.9",
        "/area 4/i",
        12,
        ["Area 44 (IFG)", "Area 4p (PreCG) right", "Area 4a (PreCG) left"],
    ),
    (
        "Superficial fiber Bundles HCP",
        re.compile("rh_SF-SF_*"),
        24,
        ["rh_SF-SF_9", "rh_SF-SF_19", "rh_SF-SF_23"],
    ),
    (
        "julich 3.0",
        siibra.get_region("julich 3.0", "hoc1 left"),
        1,
        ["Area hOc1 (V1, 17, CalcS) left"],
    ),
]


@pytest.mark.parametrize(
    "parc_spec, region_spec, result_len, check_regions", regionspecs
)
def test_find(parc_spec, region_spec, result_len, check_regions):
    parc = siibra.parcellations.get(parc_spec)
    results = parc.find(region_spec)
    assert isinstance(results, list)
    assert len(results) == result_len
    assert all(r in results for r in check_regions)


@pytest.mark.parametrize("parc, reg_spec, has_related, has_homology, has_related_ebrains_reg", [
    ("julich 2.9", "PGa", True, True, False),
    ("monkey", "PG", False, True, False),
    ("waxholm v3", "cornu ammonis 1", True, False, True),
])
def test_homologies_related_regions(parc, reg_spec, has_related, has_homology, has_related_ebrains_reg):

    reg = siibra.get_region(parc, reg_spec)
    related_assessments = [val for val in reg.get_related_regions()]
    homology_assessments = [val for val in related_assessments if val.qualification == Qualification.HOMOLOGOUS]
    other_v_assessments = [val for val in related_assessments if val.qualification == Qualification.OTHER_VERSION]

    assert has_related == (len(other_v_assessments) > 0)
    assert has_homology == (len(homology_assessments) > 0)

    if has_related_ebrains_reg:
        with ThreadPoolExecutor() as ex:
            features = ex.map(
                siibra.features.get,
                [val.assigned_structure for val in other_v_assessments],
                repeat("ebrains")
            )
        assert len([f for f in features]) > 0


def test_related_region_hemisphere():
    reg = siibra.get_region("2.9", "PGa")
    all_related_reg = [reg for reg in reg.get_related_regions()]
    assert any("left" in ass.assigned_structure.name for ass in all_related_reg)
    assert any("right" in ass.assigned_structure.name for ass in all_related_reg)


spaces = ['mni152', 'colin27']


@pytest.mark.parametrize("space", spaces)
def test_boundingbox(space):
    hoc1_l = siibra.get_region('julich', 'hoc1 left')
    hoc1_r = siibra.get_region('julich', 'hoc1 right')
    bbox_l = hoc1_l.get_boundingbox(space)
    bbox_r = hoc1_r.get_boundingbox(space)
    assert bbox_l != bbox_r, "Left and right hoc1 should not have the same bounding boxes"

    v_l = hoc1_l.get_regional_mask(space)
    v_r = hoc1_r.get_regional_mask(space)
    assert bbox_l == v_l.get_boundingbox(clip=True, background=0.0), "Boundingbox of regional mask should be the same as bounding mask of the regions"
    assert bbox_r == v_r.get_boundingbox(clip=True, background=0.0), "Boundingbox of regional mask should be the same as bounding mask of the regions"
