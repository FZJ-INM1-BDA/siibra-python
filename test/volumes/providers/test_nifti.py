from typing import Iterator, Tuple
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from siibra.core.space import Space
from siibra.locations.boundingbox import BoundingBox
from siibra.volumes.providers.nifti import NiftiProvider


@pytest.fixture(autouse=True)
def no_space_registry() -> Iterator[None]:
    # BoundingBox.transform(..., space=None) resolves the space through the
    # registry, which would load the configuration over the network.
    with patch.object(Space, "get_instance", return_value=None):
        yield


def make_provider(shape: Tuple[int, ...], affine: np.ndarray) -> NiftiProvider:
    data = np.arange(np.prod(shape), dtype="int32").reshape(shape)
    return NiftiProvider(nib.Nifti1Image(data, affine))


def assert_crop_consistent(source: nib.Nifti1Image, crop: nib.Nifti1Image) -> None:
    """Every voxel of `crop` holds the source value at the same physical position."""
    to_source = np.linalg.inv(source.affine) @ crop.affine
    for ijk in np.ndindex(crop.shape[:3]):
        src_ijk = np.rint(to_source @ [*ijk, 1])[:3].astype(int)
        np.testing.assert_array_equal(
            np.asanyarray(crop.dataobj)[ijk], np.asanyarray(source.dataobj)[tuple(src_ijk)]
        )


def test_voi_crop_with_non_unit_voxels() -> None:
    provider = make_provider((20, 20, 20), np.diag([2.0, 2.0, 2.0, 1.0]))
    crop = provider.fetch(voi=BoundingBox((10, 10, 10), (20, 20, 20), space=None))
    assert crop.shape == (5, 5, 5)
    np.testing.assert_array_equal(crop.affine[:3, 3], [10.0, 10.0, 10.0])
    assert_crop_consistent(provider.fetch(), crop)


def test_voi_crop_with_fractional_start() -> None:
    provider = make_provider((20, 20, 20), np.eye(4))
    crop = provider.fetch(voi=BoundingBox((2.5, 2.5, 2.5), (6, 6, 6), space=None))
    assert crop.shape == (4, 4, 4)
    np.testing.assert_array_equal(crop.affine[:3, 3], [2.0, 2.0, 2.0])
    assert_crop_consistent(provider.fetch(), crop)


def test_voi_crop_with_flipped_and_offset_affine() -> None:
    affine = np.array([
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    provider = make_provider((91, 109, 91), affine)
    crop = provider.fetch(voi=BoundingBox((-10, -20, 0), (10, 0, 20), space=None))
    assert crop.shape == (10, 10, 10)
    assert_crop_consistent(provider.fetch(), crop)


@pytest.mark.parametrize("voxel_size", [0.1, 0.3, 0.7, 1.2])
def test_voi_crop_on_voxel_boundaries(voxel_size: float) -> None:
    provider = make_provider((30, 30, 30), np.diag([voxel_size] * 3 + [1.0]))
    lo, hi = 7 * voxel_size, 14 * voxel_size
    crop = provider.fetch(voi=BoundingBox((lo,) * 3, (hi,) * 3, space=None))
    assert crop.shape == (7, 7, 7)
    assert_crop_consistent(provider.fetch(), crop)


def test_voi_crop_keeps_time_axis() -> None:
    provider = make_provider((10, 10, 10, 3), np.diag([2.0, 2.0, 2.0, 1.0]))
    crop = provider.fetch(voi=BoundingBox((4, 4, 4), (10, 10, 10), space=None))
    assert crop.shape == (3, 3, 3, 3)
    np.testing.assert_array_equal(crop.affine[:3, 3], [4.0, 4.0, 4.0])