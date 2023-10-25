# Copyright 2018-2021
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
"""A set of coordinates on a reference space."""

from . import location, point, boundingbox

from ..retrieval.requests import HttpRequest
from ..commons import logger

import numbers
import json
import numpy as np
from nibabel import Nifti1Image
from typing import Union


class PointSet(location.Location):
    """A set of 3D points in the same reference space,
    defined by a list of coordinates."""

    def __init__(self, coordinates, space=None, sigma_mm=0):
        """
        Construct a 3D point set in the given reference space.

        Parameters
        ----------
        coordinates : list of Point, 3-tuples or string specs
            Coordinates in mm of the given space
        space : reference space (id, name, or Space object)
            The reference space
        sigma_mm : float, or list of float
            Optional standard deviation of point locations.
        """
        location.Location.__init__(self, space)
        if isinstance(sigma_mm, numbers.Number):
            self.points = [point.Point(c, self.space, sigma_mm) for c in coordinates]
        else:
            self.points = [
                point.Point(c, self.space, s) for c, s in zip(coordinates, sigma_mm)
            ]

    def intersection(self, other: Union[location.Location, Nifti1Image]):
        """Return the subset of points that are inside the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        if isinstance(other, point.Point):
            return self if other in self else None
        elif isinstance(other, PointSet):
            return [p for p in self if p in other]
        elif isinstance(other, boundingbox.BoundingBox):
            return [p for p in self if p.contained_in(other)]
        inside = [p for p in self if p.intersects(other)]
        if len(inside) == 0:
            return None
        elif len(inside) == 1:
            return inside[0]
        else:
            return PointSet(
                [p.coordinate for p in inside],
                space=self.space,
                sigma_mm=[p.sigma for p in inside],
            )

    def intersects(self, other: Union[location.Location, Nifti1Image]):
        return len(self.intersection(other)) > 0

    @property
    def sigma(self):
        return [p.sigma for p in self]

    @property
    def has_constant_sigma(self):
        return len(set(self.sigma)) == 1

    def warp(self, space, chunksize=1000):
        """Creates a new point set by warping its points to another space"""
        from ..core.space import Space
        spaceobj = Space.get_instance(space)
        if spaceobj == self.space:
            return self
        if any(_ not in location.Location.SPACEWARP_IDS for _ in [self.space.id, spaceobj.id]):
            raise ValueError(
                f"Cannot convert coordinates between {self.space.id} and {spaceobj.id}"
            )

        src_points = self.as_list()
        tgt_points = []
        N = len(src_points)
        if N > 10e5:
            logger.info(f"Warping {N} points from {self.space.name} to {spaceobj.name} space")
        for i0 in range(0, N, chunksize):

            i1 = min(i0 + chunksize, N)
            data = json.dumps({
                "source_space": location.Location.SPACEWARP_IDS[self.space.id],
                "target_space": location.Location.SPACEWARP_IDS[spaceobj.id],
                "source_points": src_points[i0:i1]
            })
            response = HttpRequest(
                url=f"{location.Location.SPACEWARP_SERVER}/transform-points",
                post=True,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                },
                data=data,
                func=lambda b: json.loads(b.decode()),
            ).data
            tgt_points.extend(list(response["target_points"]))

        return self.__class__(coordinates=tuple(tgt_points), space=spaceobj)

    def transform(self, affine: np.ndarray, space=None):
        """Returns a new PointSet obtained by transforming the
        coordinates of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : reference space (id, name, or Space)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        return self.__class__(
            [c.transform(affine, space) for c in self.points], space
        )

    def __getitem__(self, index: int):
        if (index >= self.__len__()) or (index < 0):
            raise IndexError(
                f"Pointset has only {self.__len__()} points, "
                f"but index of {index} was requested."
            )
        else:
            return self.points[index]

    def __iter__(self):
        """Return an iterator over the coordinate locations."""
        return iter(self.points)

    def __len__(self):
        """The number of points in this PointSet."""
        return len(self.points)

    def __str__(self):
        return f"Set of points {self.space.name}: " + ", ".join(
            f"({','.join(str(v) for v in p)})" for p in self
        )

    @property
    def boundingbox(self):
        """
        Return the bounding box of these points.
        """
        from .boundingbox import BoundingBox
        XYZ = self.homogeneous[:, :3]
        sigma_min = max(self.sigma[i] for i in XYZ.argmin(0))
        sigma_max = max(self.sigma[i] for i in XYZ.argmax(0))
        return BoundingBox(
            point1=XYZ.min(0) - max(sigma_min, 1e-6),
            point2=XYZ.max(0) + max(sigma_max, 1e-6),
            space=self.space,
            sigma_mm=[sigma_min, sigma_max]
        )

    @property
    def centroid(self):
        return point.Point(self.homogeneous[:, :3].mean(0), space=self.space)

    @property
    def volume(self):
        if len(self) < 2:
            return 0
        else:
            return self.boundingbox.volume

    def as_list(self):
        """Return the point set as a list of 3D tuples."""
        return [tuple(p) for p in self]

    @property
    def homogeneous(self):
        """Access the list of 3D point as an Nx4 array of homogeneous coorindates."""
        return np.array([c.homogeneous for c in self.points]).reshape((-1, 4))
