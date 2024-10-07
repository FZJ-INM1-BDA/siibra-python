import numpy as np
import pytest
from unittest.mock import MagicMock
import nibabel as nib

from siibra.attributes.locations.pointcloud import PointCloud, unifrom_from_image
from siibra.attributes.datarecipes.volume import ImageRecipe

ptcloud = PointCloud(
    coordinates=[
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [10, 11, 12],
        [10, 11, 12],
    ],
    space_id="foo",
)

ptcld_args = [
    ([[0, 0, 0], [1, 1, 1]], None, None),
    ([(0, 0, 0), (1, 1, 1)], None, None),
    (([0, 0, 0], [1, 1, 1]), None, None),
    (((0, 0, 0), (1, 1, 1)), None, None),
    (np.array([[0, 0, 0], [1, 1, 1]]), None, None),
    ([[0, 0, 0], [1, 1, 1]], [0], AssertionError),
]


@pytest.mark.parametrize("coordinates, sigma, Error", ptcld_args)
def test_ptcloud_postinit(coordinates, sigma, Error):
    if Error:
        with pytest.raises(Error):
            PointCloud(coordinates=coordinates, sigma=sigma)
        return
    ptcld = PointCloud(coordinates=coordinates, sigma=sigma)

    assert isinstance(ptcld.coordinates, list)
    assert all(isinstance(pos, tuple) for pos in ptcld.coordinates)
    assert all(isinstance(v, (float, int)) for pos in ptcld.coordinates for v in pos)


def test_ptcloud_homogenous():
    assert ptcloud.homogeneous.tolist() == [
        [0, 0, 0, 1],
        [1, 1, 1, 1],
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
    new_ptcld = PointCloud.transform(
        ptcloud,
        [
            [1, 0, 0, 10],
            [0, 1, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ],
    )
    assert new_ptcld != ptcloud
    assert new_ptcld.coordinates == [
        (10, 20, 30),
        (11, 21, 31),
        (12, 22, 32),
        (20, 31, 42),
        (20, 31, 42),
    ]


def test_ptcloud_transform_affine_works():
    new_ptcld = PointCloud.transform(
        ptcloud,
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    assert new_ptcld != ptcloud
    assert new_ptcld.coordinates == [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2),
        (11, 10, 12),
        (11, 10, 12),
    ]


@pytest.fixture
def mock_image_provider():
    array = np.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        ],
        dtype=np.int8,
    )
    mock = MagicMock()
    mock.get_data.return_value = nib.Nifti1Image(array, affine=np.eye(4))
    yield mock


def test_uniform_from_image(mock_image_provider):
    result = unifrom_from_image(mock_image_provider)
    assert result.coordinates == [
        (1, 0, 0),
        (1, 1, 1),
        (1, 2, 2),
    ]
    assert result.sigma == [0.5, 0.5, 0.5]
