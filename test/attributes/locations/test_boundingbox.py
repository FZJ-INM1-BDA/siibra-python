import numpy as np
import pytest
from siibra.attributes.locations import boundingbox

arr0 = np.zeros((8, 8, 8), dtype=np.uint8)
arr1 = np.copy(arr0)
arr2 = np.copy(arr0)

arr0[0, 0, 0] = 1
arr0[1, 1, 1] = 1

arr1[1, 1, 1] = 1
arr1[2, 2, 2] = 1

arr2[1, 1, 1] = 1


@pytest.mark.parametrize(
    "arr, expected",
    [
        (arr0, [(0, 0, 0), (2, 2, 2)]),
        (arr1, [(1, 1, 1), (3, 3, 3)]),
        (arr2, [(1, 1, 1), (2, 2, 2)]),
    ],
)
def test_from_array(arr, expected):
    lower, higher = expected
    bbox = boundingbox.from_array(arr)
    assert tuple(bbox.minpoint) == lower
    assert tuple(bbox.maxpoint) == higher
