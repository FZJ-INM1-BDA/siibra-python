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

from . import location, boundingbox, pointset
from ..commons import logger
from ..retrieval.requests import HttpRequest

from nibabel.affines import apply_affine
from nibabel import Nifti1Image
from urllib.parse import quote
import re
import numpy as np
import json
import numbers
import hashlib
from typing import Union, Tuple


class Point(location.Location):
    """A single 3D point in reference space."""

    @staticmethod
    def parse(spec, unit="mm") -> Tuple[float, float, float]:
        """Converts a 3D coordinate specification into a 3D tuple of floats.

        Parameters
        ----------
        spec: Any of str, tuple(float,float,float)
            For string specifications, comma separation with decimal points are expected.
        unit: str
            specification of the unit (only 'mm' supported so far)
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
            digits = re.findall(pat, spec)
            if len(digits) == 3:
                return tuple(float(d) for d in digits)
        elif isinstance(spec, (tuple, list)) and len(spec) in [3, 4]:
            if len(spec) == 4:
                assert spec[3] == 1
            return tuple(float(v) for v in spec[:3])
        elif isinstance(spec, np.ndarray) and spec.size == 3:
            return tuple(float(v) for v in spec[:3])
        elif isinstance(spec, Point):
            return spec.coordinate

        raise ValueError(
            f"Cannot decode the specification {spec} (type {type(spec)}) to create a point."
        )

    def __hash__(self):
        return sum(map(hash, (self.coordinate, self.sigma, self.space.key)))

    def __init__(self, coordinatespec, space=None, sigma_mm: float = 0.0):
        """
        Construct a new 3D point set in the given reference space.

        Parameters
        ----------
        coordinatespec: 3-tuple of int/float, or string specification
            Coordinate in mm of the given space
        space: Space or str
            The reference space (id, object, or name)
        sigma_mm : float, optional
            Location uncertainty of the point

            Note
            ----
                Interpreted as the isotropic standard deviation of location.
        """
        location.Location.__init__(self, space)
        self.coordinate = Point.parse(coordinatespec)
        self.sigma = sigma_mm
        if isinstance(coordinatespec, Point):
            assert coordinatespec.sigma == sigma_mm
            assert coordinatespec.space == space

    @property
    def homogeneous(self):
        """The homogenous coordinate of this point as a 4-tuple,
        obtained by appending '1' to the original 3-tuple."""
        return self.coordinate + (1,)

    def intersection(self, other: Union[location.Location, Nifti1Image]) -> "Point":
        if isinstance(other, Point):
            return self if self == other else None
        elif isinstance(other, pointset.PointSet):
            return self if self in other else None
        elif isinstance(other, boundingbox.BoundingBox):
            return self if other.contains(self) else None
        elif isinstance(other, Nifti1Image):
            return self if self.intersects(other) else None
        else:
            raise NotImplementedError(
                f"Intersection of {self.__class__.__name__} with "
                f"{other.__class__.__name__} not implemented"
            )

    def intersects(self, other: Union[location.Location, Nifti1Image]) -> bool:
        """Returns true if this point lies in the given mask.

        NOTE: The affine matrix of the image must be set to warp voxels
        coordinates into the reference space of this Bounding Box.
        """
        # transform physical coordinates to voxel coordinates for the query
        def coordinate_inside_mask(mask, c):
            voxel = (apply_affine(np.linalg.inv(mask.affine), c) + 0.5).astype(int)
            if np.any(voxel >= mask.dataobj.shape):
                return False
            elif np.any(voxel < 0):
                return False
            elif mask.dataobj[voxel[0], voxel[1], voxel[2]] == 0:
                return False
            else:
                return True

        if isinstance(other, location.Location):
            return (self.intersection(other) is not None)
        elif isinstance(other, Nifti1Image):
            if other.ndim == 4:
                return any(
                    coordinate_inside_mask(other.slicer[:, :, :, i], self.coordinate)
                    for i in range(other.shape[3])
                )
            else:
                return coordinate_inside_mask(other, self.coordinate)
        else:
            raise NotImplementedError(
                f"Intersection test of {self.__class__.__name__} with "
                f"{other.__class__.__name__} not implemented"
            )

    def contained_in(self, other: Union[location.Location, Nifti1Image]):
        return self.intersects(other)

    def contains(self, other: Union[location.Location, Nifti1Image]):
        if isinstance(other, Point):
            return self == other
        else:
            return False

    def warp(self, space):
        """Creates a new point by warping this point to another space"""
        from ..core.space import Space
        spaceobj = Space.get_instance(space)
        if spaceobj == self.space:
            return self
        if any(_ not in location.Location.SPACEWARP_IDS for _ in [self.space.id, spaceobj.id]):
            raise ValueError(
                f"Cannot convert coordinates between {self.space.id} and {spaceobj.id}"
            )
        url = "{server}/transform-point?source_space={src}&target_space={tgt}&x={x}&y={y}&z={z}".format(
            server=location.Location.SPACEWARP_SERVER,
            src=quote(location.Location.SPACEWARP_IDS[self.space.id]),
            tgt=quote(location.Location.SPACEWARP_IDS[spaceobj.id]),
            x=self.coordinate[0],
            y=self.coordinate[1],
            z=self.coordinate[2],
        )
        response = HttpRequest(url, lambda b: json.loads(b.decode())).get()
        if any(map(np.isnan, response['target_point'])):
            logger.debug(f'Warping {str(self)} to {spaceobj.name} resulted in NaN')
            return None
        return self.__class__(
            coordinatespec=tuple(response["target_point"]), space=spaceobj.id
        )

    @property
    def volume(self):
        """ The volume of a point can be nonzero if it has a location uncertainty. """
        return self.sigma**3 * np.pi * 4. / 3.

    def __sub__(self, other):
        """Substract the coordinates of two points to get
        a new point representing the offset vector. Alternatively,
        subtract an integer from the all coordinates of this point
        to create a new one."""
        if isinstance(other, numbers.Number):
            return Point([c - other for c in self.coordinate], self.space)

        assert self.space == other.space
        return Point(
            [self.coordinate[i] - other.coordinate[i] for i in range(3)], self.space
        )

    def __lt__(self, other):
        o = other if self.space is None else other.warp(self.space)
        return all(self[i] < o[i] for i in range(3))

    def __gt__(self, other):
        o = other if self.space is None else other.warp(self.space)
        return all(self[i] > o[i] for i in range(3))

    def __eq__(self, other: 'Point'):
        if not isinstance(other, Point):
            return False
        o = other if self.space is None else other.warp(self.space)
        return all(self[i] == o[i] for i in range(3))

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other)

    def __add__(self, other):
        """Add the coordinates of two points to get
        a new point representing."""
        if isinstance(other, numbers.Number):
            return Point([c + other for c in self.coordinate], self.space)
        if isinstance(other, Point):
            assert self.space == other.space
        return Point(
            [self.coordinate[i] + other.coordinate[i] for i in range(3)], self.space
        )

    def __truediv__(self, number: float):
        """Return a new point with divided
        coordinates in the same space."""
        return Point(np.array(self.coordinate) / number, self.space, self.sigma / number)

    def __mul__(self, number: float):
        """Return a new point with multiplied
        coordinates in the same space."""
        return Point(np.array(self.coordinate) * number, self.space, self.sigma * number)

    def transform(self, affine: np.ndarray, space=None):
        """Returns a new Point obtained by transforming the
        coordinate of this one with the given affine matrix.

        Parameters
        ----------
        affine: numpy 4x4 ndarray
            affine matrix
        space: str, Space, or None
            Target reference space which is reached after applying the transform

            Note
            ----
            The consistency of this cannot be checked and is up to the user.
        """
        from ..core.space import Space
        spaceobj = Space.get_instance(space)
        x, y, z, h = np.dot(affine, self.homogeneous)
        if h != 1:
            logger.warning(f"Homogeneous coordinate is not one: {h}")
        return self.__class__((x / h, y / h, z / h), spaceobj)

    def get_enclosing_cube(self, width_mm):
        """
        Create a bounding box centered around this point with the given width.
        """
        offset = width_mm / 2
        from .boundingbox import BoundingBox
        return BoundingBox(
            point1=self - offset,
            point2=self + offset,
            space=self.space,
        )

    def __iter__(self):
        """Return an iterator over the location,
        so the Point can be easily cast to list or tuple."""
        return iter(self.coordinate)

    def __setitem__(self, index, value):
        """Write access to the coefficients of this point."""
        assert 0 <= index < 3
        values = list(self.coordinate)
        values[index] = value
        self.coordinate = tuple(values)

    def __getitem__(self, index):
        """Index access to the coefficients of this point."""
        assert 0 <= index < 3
        return self.coordinate[index]

    @property
    def boundingbox(self):
        w = max(self.sigma or 0, 1e-6)  # at least a micrometer
        return boundingbox.BoundingBox(
            self - w, self + w, self.space, self.sigma
        )

    def bigbrain_section(self):
        """
        Estimate the histological section number of BigBrain
        which corresponds to this point. If the point is given
        in another space, a warping to BigBrain space will be tried.
        """
        if self.space.id == location.Location.BIGBRAIN_ID:
            coronal_position = self[1]
        else:
            try:
                bigbrain_point = self.warp("bigbrain")
                coronal_position = bigbrain_point[1]
            except Exception:
                raise RuntimeError(
                    "BigBrain section numbers can only be determined "
                    "for points in BigBrain space, but the given point "
                    f"is given in '{self.space.name}' and could not "
                    "be converted."
                )
        return int((coronal_position + 70.0) / 0.02 + 1.5)

    @property
    def id(self) -> str:
        return hashlib.md5(
            f"{self.space.id}{','.join(str(val) for val in self)}".encode("utf-8")
        ).hexdigest()
