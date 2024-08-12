import numpy as np
from dataclasses import replace
import pytest

from siibra.attributes.locations.ops.transform import transform_point
from siibra.attributes.locations import Point

iden = np.eye(4)

scale2 = iden * 2
scale2[3,  3] = 1

transl = np.eye(4)
transl[:,3] = [3, 3, 3, 1]

ptfoo = Point(coordinate=(1., 2., 3.), space_id="foo")
ptfoo2 = Point(coordinate=(2., 4., 6.), space_id="foo")
ptfoo3 = Point(coordinate=(4., 5., 6.), space_id="foo")
ptbar = Point(coordinate=(1., 2., 3.), space_id="bar")

args = [
    (replace(ptfoo), iden, None, replace(ptfoo)),
    (replace(ptfoo), scale2, None, replace(ptfoo2)),
    (replace(ptfoo), transl, None, replace(ptfoo3)),
    (replace(ptfoo), iden, "bar", replace(ptbar)),
]


@pytest.mark.parametrize("frompt, affine, space_id, expectedpt", args)
def test_transform_point(frompt, affine, space_id, expectedpt):
    newpt = transform_point(frompt, affine, space_id)
    assert newpt == expectedpt
