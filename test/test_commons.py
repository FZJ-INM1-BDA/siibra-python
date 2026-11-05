import unittest
from unittest.mock import patch

import numpy as np
from nibabel import Nifti1Image
from parameterized import parameterized

from siibra.commons import resample_img_to_img


class TestResampleImgToImg(unittest.TestCase):
    MASK = np.array([[[0, 1], [1, 0]]], dtype="uint8")
    FLOATS = np.array([[[0., .3], [.7, 1.]]], dtype="float32")

    @parameterized.expand([
        ("mask_default", MASK, "", "nearest"),
        ("float_default", FLOATS, "", "linear"),
        ("mask_explicit", MASK, "linear", "linear"),
        ("float_explicit", FLOATS, "nearest", "nearest"),
        ("float_continuous", FLOATS, "continuous", "continuous"),
    ])
    def test_interpolation(self, _, arr, interpolation, expected):
        source = Nifti1Image(arr, np.eye(4))
        target = Nifti1Image(np.zeros_like(arr), np.eye(4))
        with patch("nilearn.image.resample_to_img") as mock_resample:
            resample_img_to_img(source, target, interpolation=interpolation)
        self.assertEqual(mock_resample.call_args.kwargs["interpolation"], expected)
