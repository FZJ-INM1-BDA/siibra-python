from siibra.attributes.dataitems.volume.image import intersect_ptcld_image, Image
from siibra.attributes.locations import PointCloud
import numpy as np
import nibabel as nib
from unittest.mock import MagicMock, patch
from itertools import product
import pytest

@pytest.fixture
def mocked_image_foo():
    with patch.object(Image, "fetch") as fetch_mock:
        image = Image(format="neuroglancer/precomputed", space_id="foo")
        yield image, fetch_mock

def test_insersect_ptcld_img(mocked_image_foo):
    dataobj = np.array(
        [
            [
                [0,0,0],
                [1,1,1],
                [0,0,0],
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,0,0],
            ],
            [
                [0,0,0],
                [0,0,0],
                [0,0,0],
            ],
        ],
        dtype=np.uint8,
    )
    image, fetch_mock = mocked_image_foo
    mock_nii = nib.nifti1.Nifti1Image(dataobj, np.identity(4))

    fetch_mock.return_value = mock_nii

    ptcld = PointCloud(
        space_id="foo",
        coordinates=list(
            product([0, 1, 2], repeat=3)
        ),
    )

    new_ptcld = intersect_ptcld_image(ptcld, image)

    assert new_ptcld is not ptcld
    assert new_ptcld != ptcld

    assert new_ptcld.space_id == ptcld.space_id

    assert new_ptcld.coordinates == [
        [0, 1, 0],
        [0, 1, 1],
        [0, 1, 2],
    ]
