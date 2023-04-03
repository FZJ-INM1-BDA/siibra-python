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
        