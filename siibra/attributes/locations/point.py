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

from dataclasses import dataclass, field, replace
from typing import Tuple, List
import numbers
import re

import numpy as np

from .base import Location
from ...commons_new.logger import logger


@dataclass
class Point(Location):
    schema = "siibra/attr/loc/point/v0.1"
    coordinate: List[float] = field(default_factory=list)
    sigma: float = 0.0

    @property
    def homogeneous(self):
        return np.atleast_2d(list(self.coordinate) + [1,])

    @staticmethod
    def transform(pt: "Point", affine: np.ndarray):
        x, y, z, h = np.dot(affine, pt.homogeneous.T)
        if h != 1:
            logger.warning(f"Homogeneous coordinate is not one: {h}")
        return replace(pt, coordinate=[x / h, y / h, z / h])

    def __add__(self, other):
        """Add the coordinates of two points to get
        a new point representing."""
        if isinstance(other, numbers.Number):
            return Point(
                coordinate=[c + other for c in self.coordinate],
                space_id=self.space_id,
            )
        if isinstance(other, Point):
            assert self.space_id == other.space_id
        return Point(
            coordinate=[self.coordinate[i] + other.coordinate[i] for i in range(3)],
            space_id=self.space_id,
            sigma=self.sigma + other.sigma,
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
        return Point(
            coordinate=[self.coordinate[i] - other.coordinate[i] for i in range(3)],
            space_id=self.space_id,
            sigma=self.sigma - other.sigma,
        )

    def __eq__(self, other: "Point"):
        if self.space_id != other.space_id:
            return False

        from . import pointset

        if isinstance(other, pointset.PointCloud):
            return other == self  # implemented at pointset
        if not isinstance(other, Point):
            return False
        return all(self[i] == other[i] for i in range(3)) and self.sigma == other.sigma

    def __iter__(self):
        """Return an iterator over the location,
        so the Point can be easily cast to list or tuple."""
        return iter(self.coordinate)


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
        return tuple(
            float(v.item()) if isinstance(v, np.ndarray) else float(v) for v in spec[:3]
        )
    elif isinstance(spec, np.ndarray) and spec.size == 3:
        return tuple(
            float(v.item()) if isinstance(v, np.ndarray) else float(v) for v in spec[:3]
        )

    raise ValueError(
        f"Cannot decode the specification {spec} (type {type(spec)}) to create a point."
    )
