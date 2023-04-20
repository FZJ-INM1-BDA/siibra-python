import pytest
import siibra

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
    ("waxholm 4", "lateral olfactory tract")
]

# Test duplicate named regions and regions with only child
@pytest.mark.parametrize("parc_spec,region_name", regions_of_concern)
def test_get_region(parc_spec,region_name):
    region = siibra.get_region(parc_spec, region_name)
    assert region
    assert region.name == region_name
