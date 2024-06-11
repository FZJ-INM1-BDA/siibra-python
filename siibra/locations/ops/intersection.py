from typing import Union
import numpy as np
from dataclasses import replace

from ..point import Pt
from ..pointset import PointCloud
from ..boundingbox import BBox
from ..base import Location
from ...exceptions import InvalidAttrCompException, UnregisteredAttrCompException
from ...commons_new.comparison import Comparison

_loc_intersection = Comparison[Location, Union[Location, None]]()


@_loc_intersection.register(Pt, Pt)
def pt_pt(pta: Pt, ptb: Pt):
    if pta.space_id != ptb.space_id:
        raise InvalidAttrCompException
    if pta.coordinate == ptb.coordinate:
        return replace(pta)


@_loc_intersection.register(Pt, PointCloud)
def pt_ptcld(pt: Pt, ptcld: PointCloud):
    if pt.space_id != ptcld.space_id:
        raise InvalidAttrCompException
    if pt.coordinate in ptcld.coordinates:
        return replace(pt)


@_loc_intersection.register(Pt, BBox)
def pt_bbox(pt: Pt, bbox: BBox):
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
    pts = [pt for pt in ptclda.to_pts() if pt_ptcld(pt, ptcldb) is not None]
    if len(pts) == 0:
        return
    if len(pts) == 1:
        return pts[0]
    return replace(ptclda, coordinates=[pt.coordinate for pt in pts])


@_loc_intersection.register(PointCloud, BBox)
def ptcld_bbox(ptcld: PointCloud, bbox: BBox):
    if ptcld.space_id != bbox.space_id:
        raise InvalidAttrCompException
    pts = [pt for pt in ptcld.to_pts() if pt_bbox(pt, bbox) is not None]
    if len(pts) == 0:
        return
    if len(pts) == 1:
        return pts[0]
    return replace(ptcld, coordinates=[pt.coordinate for pt in pts])


@_loc_intersection.register(BBox, BBox)
def bbox_bbox(bboxa: BBox, bboxb: BBox):
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
    value = _loc_intersection.get(loca, locb)
    if value is None:
        raise UnregisteredAttrCompException
    fn, switch_flag = value
    args = [locb, loca] if switch_flag else [loca, locb]
    return fn(*args)
