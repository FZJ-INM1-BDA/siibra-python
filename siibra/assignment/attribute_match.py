from typing import Type, Callable, Dict, Tuple
from functools import wraps
import numpy as np
from dataclasses import replace

from ..commons import logger
from ..concepts.attribute import Attribute
from ..exceptions import InvalidAttrCompException, UnregisteredAttrCompException
from ..descriptions.modality import Modality
from ..descriptions.regionspec import RegionSpec
from ..locations.boundingbox import BBox
from ..locations.point import Pt
from ..locations.pointset import PointCloud
from ..dataitems.image import Image


def match(attra: Attribute, attrb: Attribute) -> bool:
    typea_attr = type(attra)
    typeb_attr = type(attrb)
    key = typea_attr, typeb_attr
    if key not in COMPARE_ATTR_DICT:
        logger.debug(f"{typea_attr} and {typeb_attr} comparison has not been registered")
        raise UnregisteredAttrCompException

    fn, switch_arg = COMPARE_ATTR_DICT[key]
    args = [attrb, attra] if switch_arg else [attra, attrb]
    return fn(*args)


# TODO document me
COMPARE_ATTR_DICT: Dict[Tuple[Type[Attribute], ...], Tuple[Callable, bool]] = {}

def register_attr_comparison(attra_type: Type[Attribute], attrb_type: Type[Attribute]):
    
    def outer(fn):
        forward_key = attra_type, attrb_type
        backward_key = attrb_type, attra_type

        assert forward_key not in COMPARE_ATTR_DICT, f"{forward_key} already exist"
        assert backward_key not in COMPARE_ATTR_DICT, f"{backward_key} already exist"

        @wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)
        
        COMPARE_ATTR_DICT[forward_key] = inner, False
        COMPARE_ATTR_DICT[backward_key] = inner, True

        return inner
    return outer



@register_attr_comparison(Modality, Modality)
def compare_modality(mod1: Modality, mod2: Modality):
    return mod1.value.lower() == mod2.value.lower()

@register_attr_comparison(RegionSpec, RegionSpec)
def compare_regionspec(regspec1: RegionSpec, regspec2: RegionSpec):
    if regspec1.value == regspec2.value:
        return True
    # TODO implement fuzzy match


@register_attr_comparison(BBox, BBox)
def compare_bbox_to_bbox(bbox1: BBox, bbox2: BBox):
    if bbox2.space_id != bbox1.space_id:
        raise InvalidAttrCompException(f"bbox and image are in different space. Cannot compare the two")
    return BBox.intersect_box(bbox1, bbox2) is not None

@register_attr_comparison(Pt, BBox)
def compare_pt_to_bbox(pt: Pt, bbox: BBox):
    if bbox.space_id != pt.space_id:
        raise InvalidAttrCompException(f"bbox and image are in different space. Cannot compare the two")
    minpoint = np.array(bbox.minpoint)
    maxpoint = np.array(bbox.maxpoint)
    pt = np.array(pt.coordinate)

    return np.all(minpoint <= pt) and np.all(pt <= maxpoint) 


# TODO implement
# 
# @register_attr_comparison(BBox, Image)
# def compare_bbox_to_image(bbox: BBox, image: Image):
#     if image.space_id != bbox.space_id:
#         raise InvalidAttrCompException(f"bbox and image are in different space. Cannot compare the two.")
#     raise NotImplementedError


@register_attr_comparison(Pt, Image)
def compare_pt_to_image(pt: Pt, image: Image):
    ptcloud = PointCloud(space_id=pt.space_id, coordinates=[pt.coordinate])
    return compare_ptcloud_to_image(ptcloud=ptcloud, image=image)


@register_attr_comparison(PointCloud, Image)
def compare_ptcloud_to_image(ptcloud: PointCloud, image: Image):
    intersection = intersect_ptcld_image(ptcloud=ptcloud, image=image)
    return len(intersection.coordinates) > 0
    

def intersect_ptcld_image(ptcloud: PointCloud, image: Image) -> PointCloud:
    if image.space_id != ptcloud.space_id:
        raise InvalidAttrCompException(f"ptcloud and image are in different space. Cannot compare the two.")
    
    value_outside = 0

    img = image.data
    arr = np.asanyarray(img.dataobj)

    # transform the points to the voxel space of the volume for extracting values
    phys2vox = np.linalg.inv(img.affine)
    voxels = PointCloud.transform(ptcloud, phys2vox)
    XYZ = np.array(voxels.coordinates).astype("int")

    # temporarily set all outside voxels to (0,0,0) so that the index access doesn't fail
    # TODO in previous version, zero'th voxel is excluded on all sides (i.e. (XYZ > 0) was tested)
    # is there a reason why the zero-th voxel is excluded?
    inside = np.all((XYZ < arr.shape) & (XYZ >= 0), axis=1)
    XYZ[~inside, :] = 0

    # read out the values
    X, Y, Z = XYZ.T
    values = arr[X, Y, Z]

    # fix the outside voxel values, which might have an inconsistent value now
    values[~inside] = value_outside

    inside = list(np.where(values != value_outside)[0])
    
    return replace(ptcloud, coordinates=[ptcloud.coordinates[i] for i in inside])
