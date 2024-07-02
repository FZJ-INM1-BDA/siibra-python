import pytest


from dataclasses import replace
from typing import Type
from itertools import chain, product


from siibra.locations import Pt, BBox
from siibra.exceptions import InvalidAttrCompException
from siibra.assignment.attribute_qualification import qualify, Qualification


def bbox_x_flat(dim: int):
    point0 = Pt(space_id="foo", coordinate=[0, 0, 0])
    point1 = Pt(space_id="foo", coordinate=[100, 100, 100])
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

    return [
        BBox(space_id="foo", minpoint=minpoint.coordinate, maxpoint=maxpoint.coordinate)
        for minpoint, maxpoint in (
            (p00, p01),
            (p10, p11),
            (p20, p21),
        )
    ]


flatxbbox, flatybbox, flatzbbox = [bbox_x_flat(dim) for dim in range(3)]


def test_qualify_raise():
    bboxfoo = BBox(space_id="foo", minpoint=[0, 0, 0], maxpoint=[1, 1, 1])
    bboxbar = BBox(
        space_id="bar",
        minpoint=[0, 0, 0],
        maxpoint=[1, 1, 1],
    )
    with pytest.raises(InvalidAttrCompException):
        qualify(bboxfoo, bboxbar)


@pytest.mark.parametrize(
    "bbox1, bbox2",
    chain(
        product(flatxbbox, flatybbox),
        product(flatybbox, flatxbbox),
        product(flatxbbox, flatzbbox),
        product(flatybbox, flatzbbox),
    ),
)
def test_qualify_bbox_overlaps(bbox1: BBox, bbox2: BBox):
    assert qualify(bbox1, bbox2) == Qualification.OVERLAPS


@pytest.mark.parametrize(
    "bbox1, bbox2",
    chain(
        product(flatxbbox, flatxbbox),
        product(flatybbox, flatybbox),
        product(flatzbbox, flatzbbox),
    ),
)
def test_qualify_no_overlap(bbox1: BBox, bbox2: BBox):
    if bbox1 == bbox2:
        return
    assert qualify(bbox1, bbox2) is None


bbox_foo_5_10 = BBox(space_id="foo", minpoint=[5, 5, 5], maxpoint=[10, 10, 10])


@pytest.mark.parametrize(
    "arg1, arg2, expected, ExCls",
    [
        [bbox_foo_5_10, Pt(coordinate=(0, 0, 0), space_id="foo"), None, None],
        [bbox_foo_5_10, Pt(coordinate=(7, 0, 7), space_id="foo"), None, None],
        [bbox_foo_5_10, Pt(coordinate=(7, 15, 7), space_id="foo"), None, None],
        [
            bbox_foo_5_10,
            Pt(coordinate=(7, 7, 7), space_id="bar"),
            None,
            InvalidAttrCompException,
        ],
        [
            Pt(coordinate=(7, 7, 7), space_id="foo"),
            bbox_foo_5_10,
            Qualification.CONTAINED,
            None,
        ],
        [
            bbox_foo_5_10,
            Pt(coordinate=(7, 7, 7), space_id="foo"),
            Qualification.CONTAINS,
            None,
        ],
        [
            bbox_foo_5_10,
            Pt(coordinate=(5, 5, 5), space_id="foo"),
            Qualification.CONTAINS,
            None,
        ],
        [
            bbox_foo_5_10,
            Pt(coordinate=(10, 10, 10), space_id="foo"),
            Qualification.CONTAINS,
            None,
        ],
        [
            bbox_foo_5_10,
            Pt(coordinate=(10, 5, 10), space_id="foo"),
            Qualification.CONTAINS,
            None,
        ],
        [
            bbox_foo_5_10,
            Pt(coordinate=(10, 7, 10), space_id="foo"),
            Qualification.CONTAINS,
            None,
        ],
    ],
)
def test_compare_loc_to_loc(arg1, arg2, expected: bool, ExCls: Type[Exception]):
    if ExCls:
        with pytest.raises(ExCls):
            qualify(arg1, arg2)
        return
    assert expected == qualify(arg1, arg2)
