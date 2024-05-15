from typing import Type, Callable, Dict, Tuple
from functools import wraps

from ..commons import logger
from ..concepts.attribute import Attribute

# class AttributeMatch:

#     def match(self, attra: Attribute, attrb: Attribute) -> bool:
#         raise NotImplementedError


def match(attra: Attribute, attrb: Attribute) -> bool:
    typea_attr = type(attra)
    typeb_attr = type(attrb)
    key = typea_attr, typeb_attr
    if key not in COMPARE_ATTR_DICT:
        logger.debug(f"{typea_attr} and {typeb_attr} comparison has not been registered")
        return False
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

from ..descriptions.modality import Modality
from ..descriptions.regionspec import RegionSpec
from ..locations.boundingbox import BBox
from ..dataitems.image import Image


@register_attr_comparison(Modality, Modality)
def compare_modality(mod1: Modality, mod2: Modality):
    return mod1.name.lower() == mod2.name.lower()

@register_attr_comparison(RegionSpec, RegionSpec)
def compare_regionspec(regspec1: RegionSpec, regspec2: RegionSpec):
    if regspec1.value == regspec2.value:
        return True
    # TODO implement fuzzy match

@register_attr_comparison(BBox, Image)
def compare_bbox_to_image(bbox: BBox, image: Image):
    raise NotImplementedError

