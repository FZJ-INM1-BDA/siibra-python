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
    for idx, cmp in enumerate(props.components, start=1):
        assert cmp.volume >= props.components[idx - 1].volume




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

    v_l = hoc1_l.get_regional_map(space)
    v_r = hoc1_r.get_regional_map(space)
    assert bbox_l == v_l.get_boundingbox(clip=True, background=0.0), "Boundingbox of regional mask should be the same as bouding mask of the regions"
    assert bbox_r == v_r.get_boundingbox(clip=True, background=0.0), "Boundingbox of regional mask should be the same as bouding mask of the regions"
