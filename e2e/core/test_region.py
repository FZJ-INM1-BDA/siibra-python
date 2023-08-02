import pytest
import siibra
import re

regions = [
    siibra.get_region("julich 3.0", "Area 4p (PreCG) right"),
    siibra.get_region("julich 3.0", "hoc1 left"),
]


@pytest.mark.parametrize("region", regions)
def test_region_spatial_props(region: siibra.core.parcellation.region.Region):
    props = region.spatial_props("mni152")
    for idx, cmp in enumerate(props.components, start=1):
        assert cmp.volume >= props.components[idx - 1].volume


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
