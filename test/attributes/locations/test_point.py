import pytest
from itertools import product

from siibra.attributes.locations.point import Point


point_tuple_coord_foo = Point(coordinate=(1, 2, 3), space_id="foo")
point_list_coord_foo = Point(coordinate=[4, 4, 4], space_id="foo")
point_sum_coord_foo = Point(coordinate=[5, 6, 7], space_id="foo")


point_tuple_coord_bar = Point(coordinate=(1, 2, 3), space_id="bar")
point_list_coord_bar = Point(coordinate=[4, 4, 4], space_id="bar")
point_sum_coord_bar = Point(coordinate=[5, 6, 7], space_id="bar")


@pytest.mark.parametrize(
    "point",
    [
        point_tuple_coord_foo,
        point_list_coord_foo,
        point_sum_coord_foo,
        point_tuple_coord_bar,
        point_list_coord_bar,
        point_sum_coord_bar,
    ],
)
def test_homogenous(point: Point):
    assert point.homogeneous is not None


add_param = [
    (point_tuple_coord_foo, point_list_coord_foo, point_sum_coord_foo, None),
    (point_list_coord_foo, point_tuple_coord_foo, point_sum_coord_foo, None),
    (point_tuple_coord_foo, 1, Point(coordinate=[2, 3, 4], space_id="foo"), None),
    (point_tuple_coord_bar, point_list_coord_bar, point_sum_coord_bar, None),
    (point_list_coord_bar, point_tuple_coord_bar, point_sum_coord_bar, None),
    *list(
        product(
            (point_tuple_coord_foo, point_list_coord_foo, point_sum_coord_foo),
            (point_tuple_coord_bar, point_list_coord_bar, point_sum_coord_bar),
            (None,),
            (AssertionError,),
        )
    ),
]


@pytest.mark.parametrize("add1, add2, expected, error", add_param)
def test_add(add1, add2, expected, error):
    if error:
        with pytest.raises(error):
            _ = add1 + add2
        return
    assert add1 + add2 == expected


sub_param = [
    (point_sum_coord_foo, point_tuple_coord_foo, point_list_coord_foo, None),
    (point_sum_coord_foo, point_list_coord_foo, point_tuple_coord_foo, None),
    (point_tuple_coord_foo, 1, Point(coordinate=[0, 1, 2], space_id="foo"), None),
    (point_sum_coord_bar, point_tuple_coord_bar, point_list_coord_bar, None),
    (point_sum_coord_bar, point_list_coord_bar, point_tuple_coord_bar, None),
    *list(
        product(
            (point_tuple_coord_foo, point_list_coord_foo, point_sum_coord_foo),
            (point_tuple_coord_bar, point_list_coord_bar, point_sum_coord_bar),
            (None,),
            (AssertionError,),
        )
    ),
]


@pytest.mark.parametrize("sub1, sub2, expected, error", sub_param)
def test_sub(sub1, sub2, expected, error):
    if error:
        with pytest.raises(error):
            _ = sub1 - sub2
        return
    assert sub1 - sub2 == expected
