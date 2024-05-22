from typing import Type, Callable, Dict, Tuple, TypeVar, Generic
import numpy as np
from dataclasses import replace

from ..commons import logger, Comparison
from ..concepts.attribute import Attribute
from ..exceptions import InvalidAttrCompException, UnregisteredAttrCompException
from ..descriptions.modality import Modality
from ..descriptions.regionspec import RegionSpec
from .. import locations
from ..locations import Pt, PointCloud, BBox, intersect, DataClsLocation
from ..dataitems.image import Image


_attr_match: Comparison[Attribute, bool] = Comparison()

register_attr_comparison = _attr_match.register
def match(attra: Attribute, attrb: Attribute):
    val = _attr_match.get(attra, attrb)
    if val is None:
        logger.debug(f"{type(attra)} and {type(attrb)} comparison has not been registered")
        raise UnregisteredAttrCompException
    fn, switch_arg = val
    args = [attrb, attra] if switch_arg else [attra, attrb]
    return fn(*args)



@register_attr_comparison(Modality, Modality)
def compare_modality(mod1: Modality, mod2: Modality):
    return mod1.value.lower() == mod2.value.lower()


@register_attr_comparison(RegionSpec, RegionSpec)
def compare_regionspec(regspec1: RegionSpec, regspec2: RegionSpec):
    return regspec1.value.lower().strip() == regspec2.value.lower().strip()


@register_attr_comparison(Pt, Pt)
@register_attr_comparison(Pt, PointCloud)
@register_attr_comparison(Pt, BBox)
@register_attr_comparison(PointCloud, PointCloud)
@register_attr_comparison(PointCloud, BBox)
@register_attr_comparison(BBox, BBox)
def compare_loc_to_loc(loca: DataClsLocation, locb: DataClsLocation):
    return intersect(loca, locb) is not None


# TODO implement
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
        raise InvalidAttrCompException("ptcloud and image are in different space. Cannot compare the two.")

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
