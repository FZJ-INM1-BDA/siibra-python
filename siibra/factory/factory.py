from typing import Dict, Callable
from functools import wraps

from .. import locations
from .. import descriptions
from .. import dataitems

from ..concepts.attribute import Attribute
from ..concepts.attribute_collection import AttributeCollection
from ..concepts.feature import Feature

T = Callable[[Dict], AttributeCollection]

build_registry: Dict[str, T] = {}

def register_build_type(type_str: str):
    def outer(fn: T):
        
        @wraps(fn)
        def inner(*args, **kwargs):
            kwargs.pop("@type", None)
            return fn(*args, **kwargs)
        
        assert type_str not in build_registry, f"{type_str} already registered!"
        build_registry[type_str] = inner

        return inner
    return outer

@register_build_type(Feature.schema)
def build_feature(dict_obj):
    dict_obj.pop("@type", None)
    attribute_objs = dict_obj.pop("attributes", [])
    attributes = [attr
                  for attribute_obj in attribute_objs
                  for attr in Attribute.from_dict(attribute_obj)]
    return Feature(attributes=attributes, **dict_obj)

def build_object(dict_obj: Dict):
    schema = dict_obj.get("@type", None)
    assert schema, f"build_obj require the '@type' property of the object to be populated! {dict_obj=}"
    assert schema in build_registry, f"{schema} was not registered to be built!"
    return build_registry[schema](dict_obj)
