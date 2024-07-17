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

from ..point import Pt
from ..pointset import PointCloud, from_points
from ..boundingbox import BBox
from ..base import Location
from ....exceptions import InvalidAttrCompException
from ....commons_new.binary_op import BinaryOp

_loc_union = BinaryOp[Location, Location]()


@_loc_union.register(Pt, Pt)
def pt_pt(pta: Pt, ptb: Pt, is_switched: bool = False):
    if pta.space_id != ptb.space_id:
        raise InvalidAttrCompException
    if is_switched:
        return from_points([ptb, pta])
    else:
        return from_points([pta, ptb])


@_loc_union.register(Pt, PointCloud)
def pt_ptcld(pt: Pt, ptcld: PointCloud, is_switched: bool = False):
    if pt.space_id != ptcld.space_id:
        raise InvalidAttrCompException
    if is_switched:
        return ptcld.append(pt)
    else:
        return from_points([pt]).extend(ptcld)


@_loc_union.register(PointCloud, PointCloud)
def ptcld_ptcld(ptclda: PointCloud, ptcldb: PointCloud, is_switched: bool = False):
    if ptclda.space_id != ptcldb.space_id:
        raise InvalidAttrCompException
    if is_switched:
        return ptcldb.extend(ptclda)
    else:
        return ptclda.extend(ptcldb)


@_loc_union.register(Pt, BBox)
def pt_bbox(pt: Pt, bbox: BBox, is_switched: bool = False):
    if pt.space_id != bbox.space_id:
        raise InvalidAttrCompException
    if is_switched:
        return ptcld_bbox(bbox, from_points(pt))
    else:
        return ptcld_bbox(from_points(pt), bbox)


@_loc_union.register(PointCloud, BBox)
def ptcld_bbox(ptcld: PointCloud, bbox: BBox, is_switched: bool = False):
    if ptcld.space_id != bbox.space_id:
        raise InvalidAttrCompException
    if is_switched:
        pass  # the code is the same
    joint_cloud = ptcld
    joint_cloud.extend(bbox.corners)
    return joint_cloud.boundingbox


@_loc_union.register(BBox, BBox)
def bbox_bbox(bboxa: BBox, bboxb: BBox, is_switched: bool = False):
    if bboxa.space_id != bboxb.space_id:
        raise InvalidAttrCompException
    if is_switched:
        pass  # the code is the same
    joint_cloud = bboxa.corners
    joint_cloud.extend(bboxb.corners)
    return joint_cloud.boundingbox


def union(loca: Location, locb: Location):
    value = _loc_union.get(loca, locb)
    fn, switch_flag = value
    args = [locb, loca] if switch_flag else [loca, locb]
    return fn(*args, is_switched=switch_flag)
