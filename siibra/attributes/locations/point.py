# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Singular coordinate defined on a space, possibly with an uncertainty."""

from dataclasses import dataclass, field
from typing import Tuple, TYPE_CHECKING, Union
import numbers
import re

import numpy as np

from .base import Location


if TYPE_CHECKING:
    from nibabel import Nifti1Image


def parse_coordinate(
    spec: Union[Tuple[numbers.Number, numbers.Number, numbers.Number], np.ndarray, str],
    unit="mm",
) -> Tuple[float, float, float]:
    """
    Converts a 3D coordinate specification into a 3D tuple of floats.

    Parameters
    ----------
    spec : Union[Tuple[numbers.Number, numbers.Number, numbers.Number], np.ndarray, str]
        For string specifications, comma separation with decimal points are expected.
    unit: str, default = 'mm'
        currently, only 'mm' is supported

    Returns
    -------
    tuple(float, float, float)
    """
    if unit != "mm":
        raise NotImplementedError(
            "Coordinate parsing from strings is only supported for mm specifications so far."
        )

    if isinstance(spec, str):
        pat = r"([-\d\.]*)" + unit
        values = re.findall(pat, spec)
    elif isinstance(spec, (tuple, list)):
        return tuple(
            float(v.item()) if isinstance(v, np.ndarray) else float(v) for v in spec
        )
    elif isinstance(spec, np.ndarray):
        assert spec.size == 3
        values = spec.flatten()
    else:
        raise ValueError(
            f"Cannot decode coordinate specification {spec} (type {type(spec)}) to create a point."
        )

    if len(values) > 3:
        raise ValueError(f"Expected 3 elements, but got {len(values)}")

    return tuple(float(v) for v in values)


@dataclass
class Point(Location):
    schema = "siibra/attr/loc/point/v0.1"
    coordinate: Tuple[Union[float, int]] = field(default_factory=tuple)
    sigma: float = 0.0
    label: Union[int, float, str] = None

    def __post_init__(self):
        self.coordinate = parse_coordinate(self.coordinate)
        assert (
            len(self.coordinate) == 3
        ), f"Expected 3 elements, but got {len(self.coordinate)}"
        assert all(isinstance(coord, (float, int)) for coord in self.coordinate), (
            "Expected coordinates to be of type float, but was "
            f"{', '.join(type(coord).__name__ for coord in self.coordinate)}"
        )

    @property
    def homogeneous(self) -> np.ndarray:
        return np.atleast_2d(
            list(self.coordinate)
            + [
                1,
            ]
        )

    def to_ndarray(self) -> np.ndarray:
        """Return the coordinates as an numpy array"""
        return np.asarray(self.coordinate)

    def __add__(self, other):
        """
        Add the coordinates of two points to get a new point representing the
        elementwise addition.
        """
        if isinstance(other, numbers.Number):
            return Point(
                coordinate=[c + other for c in self.coordinate],
                space_id=self.space_id,
            )
        if isinstance(other, Point):
            assert self.space_id == other.space_id
            assert self.label == other.label
        return Point(
            coordinate=[self.coordinate[i] + other.coordinate[i] for i in range(3)],
            space_id=self.space_id,
            sigma=self.sigma + other.sigma,
            label=self.label
        )

    def __getitem__(self, index: int):
        """Index access to the coordinates of this point."""
        assert 0 <= index < 3
        return self.coordinate[index]

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            return Point(
                coordinate=[c - other for c in self.coordinate],
                space_id=self.space_id,
            )
        if isinstance(other, Point):
            assert self.space_id == other.space_id
            assert self.label == other.label
        return Point(
            coordinate=[self.coordinate[i] - other.coordinate[i] for i in range(3)],
            space_id=self.space_id,
            sigma=self.sigma - other.sigma,
            label=self.label,
        )

    def __eq__(self, other: "Point"):
        if self.space_id != other.space_id:
            return False
        try:
            if self.label != other.label:
                return False
        except TypeError:
            return False
        from . import pointcloud

        if isinstance(other, pointcloud.PointCloud):
            return other == self  # implemented at pointset
        if not isinstance(other, Point):
            return False
        return all(self[i] == other[i] for i in range(3)) and self.sigma == other.sigma

    def __iter__(self):
        """Return an iterator over dimensions of the coordinate."""
        return iter(self.coordinate)

    # TODO: profile the impact during map assignment.
    def create_gaussian_kernel(
        self, target_affine: np.ndarray, voxel_sigma_threshold: int = 3
    ) -> "Nifti1Image":
        from nibabel import Nifti1Image
        from skimage.filters import gaussian

        scaling = np.array(
            [np.linalg.norm(target_affine[:, i]) for i in range(3)]
        ).mean()
        sigma_vox = self.sigma / scaling
        voxel_transformation_affine = np.linalg.inv(target_affine)

        r = int(voxel_sigma_threshold * sigma_vox)
        k_size = 2 * r + 1
        impulse = np.zeros((k_size, k_size, k_size))
        impulse[r, r, r] = 1
        kernel = gaussian(impulse, sigma_vox)
        kernel /= kernel.sum()

        effective_r = int(kernel.shape[0] / 2)
        voxel_coords = np.round(
            Point.transform(self, voxel_transformation_affine).coordinate
        ).astype("int")
        shift = np.identity(4)
        shift[:3, -1] = voxel_coords[:3] - effective_r
        kernel_affine = np.dot(target_affine, shift)

        return Nifti1Image(dataobj=kernel, affine=kernel_affine)
