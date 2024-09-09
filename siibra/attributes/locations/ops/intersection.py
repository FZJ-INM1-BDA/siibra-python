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

from typing import Union
import numpy as np
from dataclasses import replace

from ..point import Point
from ..pointcloud import PointCloud, LabelledPointCloud
from ..boundingbox import BoundingBox
from ..base import Location
from ....exceptions import InvalidAttrCompException, UnregisteredAttrCompException
from ....commons.binary_op import BinaryOp
from ....commons.logger import logger

_loc_intersection = BinaryOp[Location, Union[Location, None]]()


@_loc_intersection.register(Point, Point)
def pt_pt(pta: Point, ptb: Point):
    if pta.space_id != ptb.space_id:
        raise InvalidAttrCompException
    if pta.coordinate == ptb.coordinate:
        return replace(pta)


@_loc_intersection.register(Point, PointCloud)
def pt_ptcld(pt: Point, ptcld: PointCloud):
    if pt.space_id != ptcld.space_id:
        raise InvalidAttrCompException
    if pt.coordinate in ptcld.coordinates:
        return replace(pt)


@_loc_intersection.register(Point, BoundingBox)
def pt_bbox(pt: Point, bbox: BoundingBox):
    if pt.space_id != bbox.space_id:
        raise InvalidAttrCompException
    minpoint = np.array(bbox.minpoint)
    maxpoint = np.array(bbox.maxpoint)
    pts = np.array(pt.coordinate)
    if np.all(minpoint <= pts) and np.all(pts <= maxpoint):
        return replace(pt)


@_loc_intersection.register(PointCloud, PointCloud)
def ptcld_ptcld(ptclda: PointCloud, ptcldb: PointCloud):
    if ptclda.space_id != ptcldb.space_id:
        raise InvalidAttrCompException
    pts = [pt for pt in ptclda.to_points() if pt_ptcld(pt, ptcldb) is not None]
    if len(pts) == 0:
        return
    if len(pts) == 1:
        return pts[0]
    return replace(
        ptclda,
        coordinates=[pt.coordinate for pt in pts],
        sigma=[pt.sigma for pt in pts],
    )


@_loc_intersection.register(LabelledPointCloud, LabelledPointCloud)
def lblptcld_lblptcld(ptclda: LabelledPointCloud, ptcldb: LabelledPointCloud):
    if ptclda.space_id != ptcldb.space_id:
        raise InvalidAttrCompException
    indices = [
        i for i, pt in enumerate(ptclda.to_points()) if pt_ptcld(pt, ptcldb) is not None
    ]
    if len(indices) == 0:
        return
    return replace(
        ptclda,
        coordinates=[ptclda.coordinates[i] for i in indices],
        labels=[ptclda.labels[i] for i in indices],
        sigma=[ptclda.sigma[i] for i in indices],
    )


@_loc_intersection.register(PointCloud, BoundingBox)
def ptcld_bbox(ptcld: PointCloud, bbox: BoundingBox):
    if ptcld.space_id != bbox.space_id:
        raise InvalidAttrCompException
    pts = [pt for pt in ptcld.to_points() if pt_bbox(pt, bbox) is not None]
    if len(pts) == 0:
        return
    if len(pts) == 1:
        return pts[0]
    return replace(
        ptcld, coordinates=[pt.coordinate for pt in pts], sigma=[pt.sigma for pt in pts]
    )


@_loc_intersection.register(LabelledPointCloud, BoundingBox)
def lblptcld_bbox(ptcld: LabelledPointCloud, bbox: BoundingBox):
    if ptcld.space_id != bbox.space_id:
        raise InvalidAttrCompException
    pts = [pt for pt in ptcld.to_points() if pt_bbox(pt, bbox) is not None]
    if len(pts) == 0:
        return
    if len(pts) == 1:
        return pts[0]
    return replace(
        ptcld,
        coordinates=[pt.coordinate for pt in pts],
        sigma=[pt.sigma for pt in pts],
        labels=[pt.label for pt in pts],
    )


@_loc_intersection.register(BoundingBox, BoundingBox)
def bbox_bbox(bboxa: BoundingBox, bboxb: BoundingBox):
    if bboxa.space_id != bboxb.space_id:
        raise InvalidAttrCompException
    minpoints = [bboxa.minpoint, bboxb.minpoint]
    maxpoints = [bboxa.maxpoint, bboxb.maxpoint]
    allpoints = minpoints + maxpoints

    result_min_coord = []
    result_max_coord = []
    for dim in range(3):
        _, A, B, _ = sorted(allpoints, key=lambda p: p[dim])
        if A in maxpoints or B in minpoints:
            return None
        result_min_coord.append(A[dim])
        result_max_coord.append(B[dim])

    return replace(bboxa, minpoint=result_min_coord, maxpoint=result_max_coord)


def intersect(loca: Location, locb: Location):
    """
    Get intersection between location A and location B. If the
    """
    value = _loc_intersection.get(loca, locb)
    if value is None:
        raise UnregisteredAttrCompException(
            f"Cannot find comparison between {type(loca)} and {type(locb)}"
        )
    fn, switch_flag = value
    if loca.space_id != locb.space_id:
        try:
            loca = loca.warp(locb.space_id)
        except Exception as e:
            logger.debug(f"Location warp error: {str(e)}")
            raise UnregisteredAttrCompException from e
    args = [locb, loca] if switch_flag else [loca, locb]
    return fn(*args)
