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

from .location import WholeBrain, Location
from .point import Point
from .pointset import PointSet
from .boundingbox import BoundingBox


def override_union(loc0: 'Location', loc1: 'Location') -> 'Location':
    """
    Add two locations of same or diffrent type to find their union as a
    Location object.

    Note
    ----
    `loc1` will be warped to `loc0` they are not in the same space.

    Parameters
    ----------
    loc0 : Location
        _description_
    loc1 : Location
        _description_

    Returns
    -------
    Location
        - Point U Point = PointSet
        - Point U PointSet = PointSet
        - PointSet U PointSet = PointSet
        - BoundingBox U BoundingBox = BoundingBox
        - BoundingBox U PointSet = BoundingBox
        - BoundingBox U Point = BoundingBox
        - WholeBrain U Location = NotImplementedError
        (all operations are commutative)
    """
    if isinstance(loc0, WholeBrain) or isinstance(loc1, WholeBrain):
        raise NotImplementedError("Union of WholeBrains is not yet implemented.")

    loc1_w = loc1.warp(loc0.space)  # adopt the space of the first location

    if isinstance(loc0, Point):  # turn Points to PointSets
        return override_union(
            PointSet([loc0], space=loc0.space, sigma_mm=loc0.sigma), loc1_w
        )

    if isinstance(loc0, PointSet):
        if isinstance(loc1_w, PointSet):
            points = set(loc0.points + loc1_w.points)
            return PointSet(
                points,
                space=loc0.space,
                sigma_mm=[p.sigma for p in points],
            )
        if isinstance(loc1_w, BoundingBox):
            return override_union(loc0.boundingbox, loc1_w)

    if isinstance(loc0, BoundingBox) and isinstance(loc1_w, BoundingBox):
        points = [loc0.minpoint, loc0.maxpoint, loc1_w.minpoint, loc1_w.maxpoint]
        return BoundingBox(
            point1=[min(p[i] for p in points) for i in range(3)],
            point2=[max(p[i] for p in points) for i in range(3)],
            space=loc0.space,
            sigma_mm=[loc0.minpoint.sigma, loc0.maxpoint.sigma]
        )

    return override_union(loc1_w, loc0)


Location.union = override_union
