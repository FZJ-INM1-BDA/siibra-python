import pytest
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

import siibra
from siibra.core.assignment import Qualification
from siibra.core.region import Region


spaces = ['mni152', 'colin27']


@pytest.mark.parametrize("space", spaces)
def test_boundingbox(space):
    hoc1_l = siibra.get_region('julich', 'hoc1 left')
    hoc1_r = siibra.get_region('julich', 'hoc1 right')
    bbox_l = hoc1_l.get_boundingbox(space)
    bbox_r = hoc1_r.get_boundingbox(space)
    assert bbox_l != bbox_r, "Left and right hoc1 should not have the same bounding boxes"

    v_l = hoc1_l.extract_map(space)
    v_r = hoc1_r.extract_map(space)
    assert bbox_l == v_l.get_boundingbox(clip=True, background=0.0), "Boundingbox of regional mask should be the same as bouding mask of the regions"
    assert bbox_r == v_r.get_boundingbox(clip=True, background=0.0), "Boundingbox of regional mask should be the same as bouding mask of the regions"
