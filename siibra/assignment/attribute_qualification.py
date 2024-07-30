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

from typing import Union, Type
from itertools import product

from .qualification import Qualification
from ..commons_new.logger import logger
from ..commons_new.binary_op import BinaryOp
from ..commons_new.string import fuzzy_match, clear_name
from ..exceptions import UnregisteredAttrCompException, InvalidAttrCompException
from ..attributes import Attribute
from ..attributes.dataitems import Image
from ..attributes.descriptions import Modality, RegionSpec, Name, ID, Facet
from ..attributes.locations import Point, PointCloud, BoundingBox, intersect
from ..cache import fn_call_cache


_attr_qual: BinaryOp[Attribute, Union[Qualification, None]] = BinaryOp()

register_attr_qualifier = _attr_qual.register


def is_qualifiable(t: Type):
    return _attr_qual.is_registered(t)


def qualify(attra: Attribute, attrb: Attribute):
    """Attempt to qualify one attribute against another attribute.

    n.b. the comparison is asymmetrical. That is, qualify(a, b) is the inverse of qualify(b, a)

    Parameters
    ----------
    attra: Attribute
    attrb: Attribute

    Returns
    -------
    bool
        If the attra matches with attrb

    Raises
    ------
    UnregisteredAttrCompException
        If the comparison of type(attra) and type(attrb) has not been registered
    InvalidAttrCompException
        If the comparison of type(attra) and type(attrb) has been registered, but the their
        value could not directly be compared (e.g. locations in different spaces)
    """
    val = _attr_qual.get(attra, attrb)
    if val is None:
        logger.debug(
            f"{type(attra)} and {type(attrb)} comparison has not been registered"
        )
        raise UnregisteredAttrCompException
    fn, switch_arg = val
    args = [attrb, attra] if switch_arg else [attra, attrb]
    result = fn(*args)

    if result is None:
        return None

    if switch_arg:
        result = result.invert()
    logger.debug(f"Comparing {attra} with {attrb}, result in {result}")
    return result


@register_attr_qualifier(ID, ID)
def qualify_id(id0: ID, id1: ID):
    if id0.value == id1.value:
        return Qualification.EXACT


@register_attr_qualifier(Name, Name)
def qualify_name(name1: Name, name2: Name):
    if name1.value == name2.value:
        return Qualification.EXACT
    if fuzzy_match(name1.value, name2.value) or fuzzy_match(name2.value, name1.value):
        return Qualification.APPROXIMATE


@register_attr_qualifier(Facet, Facet)
def qualify_aggregate_by(face_a: Facet, facet_b: Facet):
    if face_a.key == facet_b.key and face_a.value == facet_b.value:
        return Qualification.EXACT


@register_attr_qualifier(Modality, Modality)
def qualify_modality(mod1: Modality, mod2: Modality):
    if mod1.value == mod2.value:
        return Qualification.EXACT

    if mod1.value.lower() == mod2.value.lower():
        return Qualification.APPROXIMATE


@register_attr_qualifier(RegionSpec, RegionSpec)
@fn_call_cache
def qualify_regionspec(regspec1: RegionSpec, regspec2: RegionSpec):

    # if both parcellation_ids are present, short curcuit if parc id do not match
    if (
        regspec1.parcellation_id is not None
        and regspec2.parcellation_id is not None
        and regspec1.parcellation_id != regspec2.parcellation_id
    ):
        raise InvalidAttrCompException(
            f"{regspec1.parcellation_id=!r} != {regspec2.parcellation_id=!r}"
        )

    for region1, region2 in product(regspec1.decode(), regspec2.decode()):
        if region1 in region2.ancestors:
            return Qualification.CONTAINS
        if region2 in region1.ancestors:
            return Qualification.CONTAINED
        if region1 == region2:
            return Qualification.EXACT

    cleaned_name1 = clear_name(regspec1.value)
    cleaned_name2 = clear_name(regspec2.value)
    if fuzzy_match(cleaned_name1, cleaned_name2):
        return Qualification.APPROXIMATE


@register_attr_qualifier(Point, Point)
def qualify_pt_to_pt(pt1: Point, pt2: Point):
    if intersect(pt1, pt2):
        return Qualification.EXACT


@register_attr_qualifier(Point, PointCloud)
@register_attr_qualifier(Point, BoundingBox)
def qualify_pt_ptcld_bbox(pt: Point, ptcld_bbox: Union[PointCloud, BoundingBox]):
    if intersect(pt, ptcld_bbox):
        return Qualification.CONTAINED


@register_attr_qualifier(PointCloud, BoundingBox)
def qualify_ptcld_bbox(ptcld: PointCloud, bbox: BoundingBox):
    intersected = intersect(ptcld, bbox)
    if not intersected:
        return None

    num_pts = -1
    if isinstance(intersected, Point):
        num_pts = 1
    if isinstance(intersected, PointCloud):
        num_pts = len(intersected.coordinates)
    if num_pts < 0:
        raise TypeError("ptcld and bbox intersected must beeither pt or ptcloud")

    if num_pts < len(ptcld.coordinates):
        return Qualification.OVERLAPS
    if num_pts == len(ptcld.coordinates):
        return Qualification.CONTAINED

    raise RuntimeError(
        "intersection of ptcloud to bbox resulted in pointcloud with points"
        "larger than src ptcloud"
    )


@register_attr_qualifier(PointCloud, PointCloud)
def qualify_ptcld_ptcld(ptclda: PointCloud, ptcldb: PointCloud):
    intersected = intersect(ptclda, ptcldb)
    if intersected is None:
        return None

    num_pts = -1

    if isinstance(intersected, Point):
        num_pts = 1
    if isinstance(intersected, PointCloud):
        num_pts = len(intersected.coordinates)
    if num_pts < 0:
        raise TypeError("ptcld and ptcld intersected must beeither pt or ptcloud")

    numa_pts = len(ptclda.coordinates)
    numb_pts = len(ptcldb.coordinates)

    if num_pts == numa_pts == numb_pts:
        return Qualification.EXACT

    assert num_pts <= min(
        numa_pts, numb_pts
    ), "Intersect pts must be less or equal to src ptclouds"

    if num_pts == numa_pts:
        return Qualification.CONTAINED

    if num_pts == numb_pts:
        return Qualification.CONTAINS

    return Qualification.OVERLAPS


@register_attr_qualifier(BoundingBox, BoundingBox)
def qualify_bbox_bbox(bboxa: BoundingBox, bboxb: BoundingBox):
    intersected = intersect(bboxa, bboxb)
    if intersected is None:
        return None
    if not isinstance(intersected, BoundingBox):
        raise TypeError(
            f"intersection of bboxes should be a bbox, but is a {type(intersected)} instead!"
        )

    minpt_is_a, maxpt_is_a = (
        bboxa._minpoint == intersected._minpoint,
        bboxa._maxpoint == intersected._maxpoint,
    )
    minpt_is_b, maxpt_is_b = (
        bboxb._minpoint == intersected._minpoint,
        bboxb._maxpoint == intersected._maxpoint,
    )
    if all((minpt_is_a, maxpt_is_a, minpt_is_b, maxpt_is_b)):
        return Qualification.EXACT

    if minpt_is_a and maxpt_is_a:
        return Qualification.CONTAINED
    if minpt_is_b and maxpt_is_b:
        return Qualification.CONTAINS

    return Qualification.OVERLAPS


@register_attr_qualifier(PointCloud, Image)
def qualify_ptcld_image(ptcld: PointCloud, image: Image):
    intersected = intersect(ptcld, image)
    if isinstance(intersected, PointCloud):
        if len(intersected.coordinates) == len(ptcld.coordinates):
            return Qualification.CONTAINED
        return Qualification.OVERLAPS
    if isinstance(intersected, Point):
        return Qualification.OVERLAPS


# TODO implement
# @register_attr_qualifier(BBox, Image)
# def compare_bbox_to_image(bbox: BBox, image: Image):
#     if image.space_id != bbox.space_id:
#         raise InvalidAttrCompException(f"bbox and image are in different space. Cannot compare the two.")
#     raise NotImplementedError
