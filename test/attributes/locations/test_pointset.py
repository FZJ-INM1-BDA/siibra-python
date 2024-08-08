import numpy as np

from siibra.attributes.locations import PointCloud

ptcloud = PointCloud(
    coordinates=[
        [0, 0, 0],
        [1, 1 ,1],
        [2, 2, 2],
        [10, 11, 12],
        [10, 11, 12],
    ],
    space_id="foo"
)

def test_ptcloud_homogenous():
    assert ptcloud.homogeneous.tolist() == [
        [0, 0, 0, 1],
        [1, 1 ,1, 1],
        [2, 2, 2, 1],
        [10, 11, 12, 1],
        [10, 11, 12, 1],
    ]

def test_ptcloud_transform_new_obj():
    new_ptcld = PointCloud.transform(ptcloud, np.identity(4))
    assert new_ptcld is not ptcloud


def test_ptcloud_transform_value_eql():
    new_ptcld = PointCloud.transform(ptcloud, np.identity(4))
    assert new_ptcld == ptcloud

def test_ptcloud_transform_value_not_eql():
    new_ptcld = PointCloud.transform(ptcloud, [
        [1, 0, 0, 10],
        [0, 1, 0, 20],
        [0, 0, 1, 30],
        [0, 0, 0, 1],
    ])
    assert new_ptcld != ptcloud
    assert new_ptcld.coordinates == [
        [10, 20, 30],
        [11, 21, 31],
        [12, 22, 32],
        [20, 31, 42],
        [20, 31, 42],
    ]


def test_ptcloud_transform_affine_works():
    new_ptcld = PointCloud.transform(ptcloud, [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    assert new_ptcld != ptcloud
    assert new_ptcld.coordinates == [
        [0, 0, 0],
        [1, 1 ,1],
        [2, 2, 2],
        [11, 10, 12],
        [11, 10, 12],
    ]
