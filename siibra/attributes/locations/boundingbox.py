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

"""A box defined by two farthest corner coordinates on a specific space."""

from itertools import product
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .base import Location
from . import point, pointset
from ...commons_new.logger import logger


@dataclass
class BoundingBox(Location):
    schema = "siibra/attr/loc/boundingbox/v0.1"
    minpoint: List[float] = field(default_factory=list)
    maxpoint: List[float] = field(default_factory=list)
    space_id: str = None

    def __post_init__(self):
        # TODO: correctly parse sigma vals
        self._minpoint = point.Point(coordinate=self.minpoint, space_id=self.space_id)
        self._maxpoint = point.Point(coordinate=self.maxpoint, space_id=self.space_id)

    def __eq__(self, other: 'BoundingBox'):
        if not isinstance(other, BoundingBox):
            return False
        return self.minpoint == other.minpoint and self.maxpoint == other.maxpoint

    @property
    def volume(self) -> float:
        """The volume of the boundingbox in mm^3"""
        return np.prod(self.shape)

    @property
    def center(self) -> 'point.Point':
        return self._minpoint + (self._maxpoint - self._minpoint) / 2

    @property
    def shape(self):
        """The distances of the diagonal points in each axis. (Accounts for sigma)."""
        return tuple(
            (self._maxpoint + self._maxpoint.sigma)
            - (self._minpoint - self._minpoint.sigma)
        )

    @property
    def corners(self):
        """
        Returns all 8 corners of the box as a pointset.
        Note
        ----
        x0, y0, z0 = self.minpoint
        x1, y1, z1 = self.maxpoint
        all_corners = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x0, y1, z0),
            (x1, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x0, y1, z1),
            (x1, y1, z1)
        ]
        TODO: deal with sigma. Currently, returns the mean of min and max point.
        """
        xs, ys, zs = zip(
            self.minpoint,
            self.maxpoint
        )
        return pointset.PointCloud(coordinates=[
            [x, y, z]
            for x, y, z in product(xs, ys, zs)
        ], space_id=self.space_id
        )


def estimate_affine(bbox: BoundingBox, space):
    """
    Computes an affine transform which approximates
    the nonlinear warping of the eight corner points
    to the desired target space.
    The transform is estimated using a least squares
    solution to A*x = b, where A is the matrix of
    point coefficients in the space of this bounding box,
    and b are the target coefficients in the given space
    after calling the nonlinear warping.
    """

    x0, y0, z0 = bbox.minpoint
    x1, y1, z1 = bbox.maxpoint

    # set of 8 corner points in source space
    corners1 = pointset.PointCloud(
        coordinates=[
            (x0, y0, z0),
            (x0, y0, z1),
            (x0, y1, z0),
            (x0, y1, z1),
            (x1, y0, z0),
            (x1, y0, z1),
            (x1, y1, z0),
            (x1, y1, z1)
        ],
        space_id=bbox.space_id
    )

    # coefficient matrix from original points
    A = np.hstack(
        [
            [np.kron(np.eye(3), np.r_[tuple(c), 1])]
            for c in corners1
        ]).squeeze()

    # righthand side from warped points
    corners2 = corners1.warp(space)
    b = np.hstack(corners2.as_list())

    # least squares solution
    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    affine = np.vstack([x.reshape((3, 4)), np.array([0, 0, 0, 1])])

    # test
    errors = []
    for c1, c2 in zip(list(corners1), list(corners2)):
        errors.append(
            np.linalg.norm(
                np.dot(affine, np.r_[tuple(c1), 1])
                - np.r_[tuple(c2), 1]
            )
        )
    logger.debug(
        f"Average projection error under linear approximation "
        f"was {np.mean(errors):.2f} pixel"
    )

    return affine


def _determine_bounds(array: np.ndarray, threshold=0):
    """
    Bounding box of nonzero values in a 3D array.
    https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    x = np.any(array > threshold, axis=(1, 2))
    y = np.any(array > threshold, axis=(0, 2))
    z = np.any(array > threshold, axis=(0, 1))
    nzx, nzy, nzz = [np.where(v) for v in (x, y, z)]
    if any(len(nz[0]) == 0 for nz in [nzx, nzy, nzz]):
        # empty array
        return None
    xmin, xmax = nzx[0][[0, -1]]
    ymin, ymax = nzy[0][[0, -1]]
    zmin, zmax = nzz[0][[0, -1]]
    return np.array([[xmin, xmax + 1], [ymin, ymax + 1], [zmin, zmax + 1], [1, 1]])


def from_array(array: np.ndarray, threshold=0, space_id: str = None) -> "BoundingBox":
    """
    Find the bounding box of an array.

    Parameters
    ----------
    array : np.ndarray
    threshold : int, default: 0
    space : Space, default: None
    """
    bounds = _determine_bounds(array, threshold)
    return BoundingBox(minpoint=bounds[:3, 0], maxpoint=bounds[:3, 1], space_id=space_id)
