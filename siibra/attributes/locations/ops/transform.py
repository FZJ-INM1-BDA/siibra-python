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

from typing import Generic, TypeVar, Dict, Callable
from functools import wraps
from dataclasses import replace

import numpy as np

from ..base import Location
from ..point import Point
from ..pointcloud import PointCloud
from ..boundingbox import BoundingBox
from ....commons_new.logger import logger

T = TypeVar("T", bound=Location)
_tranformers: Dict["Location", Callable] = {}


def _register_warper(location_type: Generic[T]):

    def outer(fn: Callable[[Location], Location]):
        _tranformers[location_type] = fn
        return fn

    return outer


@_register_warper(Point)
def transform_point(
    point: Point, affine: np.ndarray, target_space_id: str = None
) -> Point:
    """Returns a new Point obtained by transforming the
    coordinate of this one with the given affine matrix.
    TODO this needs to maintain sigma

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
    if point.sigma != 0:
        logger.warning("NotYetImplemented: sigma won't be retained.")
    x, y, z, h = np.dot(affine, point.homogeneous.T)
    if h != 1:
        logger.warning(f"Homogeneous coordinate is not one: {h}")
    return replace(point, coordinate=(x / h, y / h, z / h), space_id=target_space_id)


@_register_warper(PointCloud)
def transform_pointcloud(
    ptcloud: PointCloud, affine: np.ndarray, target_space_id: str = None
) -> PointCloud:
    if any(s != 0 for s in ptcloud.sigma):
        logger.warning("NotYetImplemented: sigma won't be retained.")
    return replace(
        ptcloud,
        coordinates=PointCloud._parse_values(
            np.dot(affine, ptcloud.homogeneous.T)[:3, :].T
        )[0],
        space_id=target_space_id,
    )


@_register_warper(BoundingBox)
def transform_boundingbox(
    bbox: BoundingBox, affine: np.ndarray, target_space_id: str = None
) -> BoundingBox:
    tranformed_corners = bbox.corners.transform(affine, target_space_id)
    return tranformed_corners.boundingbox


def transform(loc: T, affine: np.ndarray, space_id: str = None) -> T:
    return _tranformers[type(loc)](loc, affine, space_id or loc.space_id)
