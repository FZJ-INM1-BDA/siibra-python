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

import requests
from urllib.parse import quote
import json
from typing import TypeVar, Dict, Callable, Type

import numpy as np

from ..base import Location
from ..point import Point
from ..pointset import PointCloud
from ..boundingbox import BoundingBox
from ....cache import fn_call_cache
from ....commons_new.logger import logger
from ....exceptions import SpaceWarpingFailedError

SPACEWARP_SERVER = "https://hbp-spatial-backend.apps.hbp.eu/v1"

# lookup of space identifiers to be used by SPACEWARP_SERVER
SPACEWARP_IDS = {
    "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2": "MNI 152 ICBM 2009c Nonlinear Asymmetric",
    "minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992": "MNI Colin 27",
    "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588": "Big Brain (Histology)",
}

T = TypeVar("T", Location)
_warpers: Dict["Location", Callable] = {}
session = requests.Session()


def _register_warper(location_type: Type):
    def outer(fn):
        assert (
            location_type not in _warpers
        ), f"{location_type.__name__} has already been registereed"
        _warpers[location_type] = fn
        return fn

    return outer


@_register_warper(Point)
@fn_call_cache
def warp_point(point: Point, target_space_id: str) -> Point:
    if target_space_id == point.space_id:
        return point
    if any(_id not in SPACEWARP_IDS for _id in [point.space_id, target_space_id]):
        raise ValueError(
            f"Cannot convert coordinates between {point.space_id} and {target_space_id}"
        )
    url = "{server}/transform-point?source_space={src}&target_space={tgt}&x={x}&y={y}&z={z}".format(
        server=SPACEWARP_SERVER,
        src=quote(SPACEWARP_IDS[point.space_id]),
        tgt=quote(SPACEWARP_IDS[target_space_id]),
        x=point.coordinate[0],
        y=point.coordinate[1],
        z=point.coordinate[2],
    )
    resp = session.get(url)
    resp.raise_for_status()
    response = resp.json()
    if any(map(np.isnan, response["target_point"])):
        raise SpaceWarpingFailedError(
            f"Warping {str(point)} to {SPACEWARP_IDS[target_space_id]} resulted in 'NaN'"
        )

    return Point(coordinate=tuple(response["target_point"]), space_id=target_space_id)


@_register_warper(PointCloud)
@fn_call_cache
def warp_pointcloud(ptcloud: PointCloud, target_space_id: str) -> PointCloud:
    chunksize = 1000
    if target_space_id == ptcloud.space_id:
        return ptcloud
    if any(_id not in SPACEWARP_IDS for _id in [ptcloud.space_id, target_space_id]):
        raise ValueError(
            f"Cannot convert coordinates between {ptcloud.space_id} and {target_space_id}"
        )

    src_points = ptcloud.coordinates
    tgt_points = []
    numofpoints = len(src_points)
    if numofpoints > 10e5:
        logger.info(
            f"Warping {numofpoints} points from {ptcloud.space.name} to {SPACEWARP_IDS[target_space_id]} space"
        )
    for i0 in range(0, numofpoints, chunksize):

        i1 = min(i0 + chunksize, numofpoints)
        data = json.dumps(
            {
                "source_space": SPACEWARP_IDS[ptcloud.space_id],
                "target_space": SPACEWARP_IDS[target_space_id],
                "source_points": src_points[i0:i1],
            }
        )
        req = session.post(
            url=f"{SPACEWARP_SERVER}/transform-points",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            data=data,
        )
        req.raise_for_status()
        response = req.json()
        if np.any(np.isnan(response["target_points"])):
            raise SpaceWarpingFailedError(
                f"Warping PointCloud to {SPACEWARP_IDS[target_space_id]} resulted in 'NaN'"
            )

        tgt_points.extend(list(response["target_points"]))

    return PointCloud(coordinates=tuple(tgt_points), space_id=target_space_id)


@_register_warper(BoundingBox)
def warp_boundingbox(bbox: BoundingBox, space_id: str) -> BoundingBox:
    corners = bbox.corners
    corners_warped = warp_pointcloud(corners, target_space_id=space_id)
    return corners_warped.boundingbox


def warp(loc: T, space_id: str) -> T:
    return _warpers[type(loc)](loc, space_id)
