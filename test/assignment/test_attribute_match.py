import pytest
from unittest.mock import MagicMock, patch

from dataclasses import replace
from typing import Tuple, Union, Type
from itertools import chain, product
import numpy as np
import nibabel as nib


from siibra.locations import PointCloud, Pt, BBox
from siibra.exceptions import InvalidAttrCompException, UnregisteredAttrCompException
from siibra.assignment.attribute_match import compare_bbox_to_bbox, intersect_ptcld_image, compare_pt_to_bbox, match
import siibra.assignment.attribute_match


def bbox_x_flat(dim: int):
    point0 = Pt(space_id="foo", coordinate=[0, 0 ,0])
    point1 = Pt(space_id="foo", coordinate=[100, 100 ,100])
    point1.coordinate[dim] = 1

    p00 = replace(point0)
    p01 = replace(point1)
    
    p10 = replace(point0)
    p10.coordinate[dim] += 5 
    p11 = replace(point1)
    p11.coordinate[dim] += 5
    
    p20 = replace(point0)
    p20.coordinate[dim] += 5 
    p21 = replace(point1)
    p21.coordinate[dim] += 5 

    return [BBox(space_id="foo",
            minpoint=minpoint.coordinate,
            maxpoint=maxpoint.coordinate)
            for minpoint, maxpoint in ((p00, p01), (p10, p11),(p20, p21),)]

flatxbbox, flatybbox, flatzbbox = [bbox_x_flat(dim) for dim in range(3)]

def test_compare_bbox_to_bbox_raise():
    bboxfoo = BBox(space_id="foo",
                   minpoint=[0, 0, 0],
                   maxpoint=[1, 1, 1])
    bboxbar = BBox(space_id="bar",
                   minpoint=[0, 0, 0],
                   maxpoint=[1, 1, 1],)
    with pytest.raises(InvalidAttrCompException) as e:
        compare_bbox_to_bbox(bboxfoo, bboxbar)

@pytest.mark.parametrize("bbox1, bbox2",
                         chain(
                             product(flatxbbox, flatybbox),
                             product(flatybbox, flatxbbox),
                             product(flatxbbox, flatzbbox),
                             product(flatybbox, flatzbbox),
                         ))
def test_compare_bbox_to_bbox_true(bbox1: BBox, bbox2: BBox):
    assert compare_bbox_to_bbox(bbox1, bbox2)


@pytest.mark.parametrize("bbox1, bbox2",
                         chain(
                             product(flatxbbox, flatxbbox),
                             product(flatybbox, flatybbox),
                             product(flatzbbox, flatzbbox),
                         ))
def test_compare_bbox_to_bbox_false(bbox1: BBox, bbox2: BBox):
    if bbox1 == bbox2:
        return
    assert not compare_bbox_to_bbox(bbox1, bbox2)


bbox_foo_5_10 = BBox(space_id="foo", minpoint=[5, 5, 5], maxpoint=[10, 10, 10])

@pytest.mark.parametrize("bbox, pt, expected, ExCls", [
    [bbox_foo_5_10, Pt(coordinate=(0, 0, 0), space_id="foo"), False, None ],
    [bbox_foo_5_10, Pt(coordinate=(7, 0, 7), space_id="foo"), False, None ],
    [bbox_foo_5_10, Pt(coordinate=(7, 15, 7), space_id="foo"), False, None ],
    [bbox_foo_5_10, Pt(coordinate=(7, 7, 7), space_id="bar"), None, InvalidAttrCompException ],
    [bbox_foo_5_10, Pt(coordinate=(7, 7, 7), space_id="foo"), True, None ],
    [bbox_foo_5_10, Pt(coordinate=(5, 5, 5), space_id="foo"), True, None ],
    [bbox_foo_5_10, Pt(coordinate=(10, 10, 10), space_id="foo"), True, None ],
    [bbox_foo_5_10, Pt(coordinate=(10, 5, 10), space_id="foo"), True, None ],
    [bbox_foo_5_10, Pt(coordinate=(10, 7, 10), space_id="foo"), True, None ],
    
])
def test_compare_pt_to_bbox(bbox: BBox, pt: Pt, expected: bool, ExCls: Type[Exception]):
    if ExCls:
        with pytest.raises(ExCls):
            compare_pt_to_bbox(pt, bbox)
        return
    assert expected == compare_pt_to_bbox(pt, bbox)


def test_insersect_ptcld_img():
    dataobj = np.array([[
        [ 0, 0, 0,],
        [ 1, 1, 1,],
        [ 0, 0, 0,],
    ]], dtype=np.uint8)

    mock_nii = nib.nifti1.Nifti1Image(dataobj, np.identity(4))

    image = MagicMock()

    image.data = mock_nii
    image.space_id = "foo"

    ptcld = PointCloud(space_id="foo",
                       coordinates=[[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                    [2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 2], ])
    
    new_ptcld = intersect_ptcld_image(ptcld, image)

    assert new_ptcld is not ptcld
    assert new_ptcld != ptcld

    assert new_ptcld.space_id == ptcld.space_id
    assert new_ptcld.sigma == ptcld.sigma

    assert new_ptcld.coordinates == [
        [0, 1, 0],
    ]

@patch("siibra.assignment.attribute_match.COMPARE_ATTR_DICT", dict())
def test_match_not_in_compare_dict_miss():
    with pytest.raises(UnregisteredAttrCompException):
        match(0, 1)

cmp_fn = MagicMock()

@patch("siibra.assignment.attribute_match.COMPARE_ATTR_DICT", {
    (int, int): (cmp_fn, False)
})
def test_match_not_in_compare_dict_hit_false():
    cmp_fn.reset_mock()
    cmp_fn.return_value = True
    assert match(0, 1)
    cmp_fn.assert_called_once_with(0, 1)


@patch("siibra.assignment.attribute_match.COMPARE_ATTR_DICT", {
    (int, int): (cmp_fn, True)
})
def test_match_not_in_compare_dict_hit_true():
    cmp_fn.reset_mock()
    cmp_fn.return_value = True
    assert match(0, 1)
    cmp_fn.assert_called_once_with(1, 0)

