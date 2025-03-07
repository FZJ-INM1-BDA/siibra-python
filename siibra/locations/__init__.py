# Copyright 2018-2025
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
"""Handles spatial concepts and spatial operation like warping between spaces."""
from typing import Iterable
from functools import reduce

from .location import Location
from .point import Point
from .pointcloud import PointCloud, Contour, from_points
from .experimental import AxisAlignedPatch, Plane
from .boundingbox import BoundingBox


def reassign_union(*args: Iterable["Location"]) -> "Location":
    return reduce(pairwise_union, args)


def pairwise_union(loc0: "Location", loc1: "Location") -> "Location":
    """
    Add two locations of same or different type to find their union as a
    Location object.

    Note
    ----
    `loc1` will be warped to `loc0` they are not in the same space.

    Parameters
    ----------
    loc0 : Location
    loc1 : Location

    Returns
    -------
    Location
        - Point U Point = PointCloud
        - Point U PointCloud = PointCloud
        - PointCloud U PointCloud = PointCloud
        - BoundingBox U BoundingBox = BoundingBox
        - BoundingBox U PointCloud = BoundingBox
        - BoundingBox U Point = BoundingBox
        - WholeBrain U Location = NotImplementedError
        (all operations are commutative)
    """
    if loc0 is None or loc1 is None:
        return loc0 or loc1

    # All location types should be unionable among each other and this should
    # be implemented here to avoid code repetition. Volumes are the only type of
    # location that has its own union method since it is not a part of locations
    # module and to avoid importing Volume here.
    if not all(
        isinstance(loc, (Point, PointCloud, BoundingBox)) for loc in [loc0, loc1]
    ):
        try:
            return loc1.union(loc0)
        except Exception:
            raise NotImplementedError(
                f"There are no union method for {(loc0.__class__.__name__, loc1.__class__.__name__)}"
            )

    # convert Points to PointClouds
    loc0, loc1 = [
        from_points([loc]) if isinstance(loc, Point) else loc for loc in [loc0, loc1]
    ]

    # adopt the space of the first location
    loc1_w = loc1.warp(loc0.space)

    if isinstance(loc0, PointCloud):
        if isinstance(loc1_w, PointCloud):
            points = list(dict.fromkeys([*loc0, *loc1_w]))
            return from_points(points)
        if isinstance(loc1_w, BoundingBox):
            return reassign_union(loc0.boundingbox, loc1_w)

    if isinstance(loc0, BoundingBox) and isinstance(loc1_w, BoundingBox):
        coordinates = [loc0.minpoint, loc0.maxpoint, loc1_w.minpoint, loc1_w.maxpoint]
        return BoundingBox(
            point1=[min(p[i] for p in coordinates) for i in range(3)],
            point2=[max(p[i] for p in coordinates) for i in range(3)],
            space=loc0.space,
            sigma_mm=[loc0.minpoint.sigma, loc0.maxpoint.sigma],
        )

    return reassign_union(loc1_w, loc0)


Location.union = reassign_union
