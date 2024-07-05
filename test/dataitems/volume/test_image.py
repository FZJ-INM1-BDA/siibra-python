from siibra.dataitems.volume.image import intersect_ptcld_image
from siibra.locations import PointCloud
import numpy as np
import nibabel as nib
from unittest.mock import MagicMock


def test_insersect_ptcld_img():
    dataobj = np.array(
        [
            [
                [
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    0,
                    0,
                ],
            ]
        ],
        dtype=np.uint8,
    )

    mock_nii = nib.nifti1.Nifti1Image(dataobj, np.identity(4))

    image = MagicMock()

    image.fetch.return_value = mock_nii
    image.space_id = "foo"

    ptcld = PointCloud(
        space_id="foo",
        coordinates=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
        ],
    )

    new_ptcld = intersect_ptcld_image(ptcld, image)

    assert new_ptcld is not ptcld
    assert new_ptcld != ptcld

    assert new_ptcld.space_id == ptcld.space_id
    assert new_ptcld.sigma == ptcld.sigma

    assert new_ptcld.coordinates == [
        [0, 1, 0],
    ]
