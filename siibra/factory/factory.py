from typing import Dict, Callable
from functools import wraps

from .. import locations
from .. import descriptions
from .. import dataitems

from ..concepts.attribute import Attribute
from ..concepts.attribute_collection import AttributeCollection
from ..concepts.feature import Feature
from ..atlases import region, parcellation
from ..descriptions import Name, SpeciesSpec, ID
from ..commons import create_key

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
def build_feature(dict_obj: dict):
    dict_obj.pop("@type", None)
    attribute_objs = dict_obj.pop("attributes", [])
    attributes = tuple(
        att
        for attribute_obj in attribute_objs
        for att in Attribute.from_dict(attribute_obj)
    )
    return Feature(attributes=attributes, **dict_obj)


@register_build_type(region.Region.schema)
def build_region(dict_obj: dict, parc_id: ID = None, species: SpeciesSpec = None):
    """_summary_

    Parameters
    ----------
    dict_obj : dict
        see Region.schema
    parc_id : ID, optional
        Either dict_obj has an ID attribute in attributes field or a
        parcellation ID is required.
    species : SpeciesSpec, optional
        Either dict_obj has an SpeciesSpec attribute in attributes field or a
        parcellation SpeciesSpec is required.

    Returns
    -------
    region.Region
    """
    dict_obj.pop("@type", None)
    attribute_objs = dict_obj.pop("attributes", [])
    attributes = tuple(
        att
        for attribute_obj in attribute_objs
        for att in Attribute.from_dict(attribute_obj)
    )
    if ID not in filter(lambda a: isinstance(a, ID), attributes):
        name = next(iter(filter(lambda a: isinstance(a, Name), attributes))).value
        attributes += (ID(value=f"{parc_id.value}_{create_key(name)}"),)
    if SpeciesSpec not in filter(lambda a: isinstance(a, ID), attributes):
        attributes += (species,)
    return region.Region(
        attributes=attributes,
        children=tuple(map(
            lambda r: build_region(r, parc_id, species),
            dict_obj.get("children", [])
        ))
    )


@register_build_type(parcellation.Parcellation.schema)
def build_parcellation(dict_obj: dict):
    dict_obj.pop("@type", None)
    attribute_objs = dict_obj.pop("attributes", [])
    attributes = tuple(
        att
        for attribute_obj in attribute_objs
        for att in Attribute.from_dict(attribute_obj)
    )
    parc_id = next(iter(filter(lambda a: isinstance(a, ID), attributes)))
    assert parc_id, "A parcellation must have an ID attribute."
    species = next(iter(filter(lambda a: isinstance(a, SpeciesSpec), attributes)))
    assert parc_id, "A parcellation must have a SpeciesSpec attribute."
    return parcellation.Parcellation(
        attributes=attributes,
        children=tuple(map(
            lambda r: build_region(r, parc_id, species),
            dict_obj.get("regions", [])
        ))
    )


def build_object(dict_obj: Dict):
    schema = dict_obj.get("@type", None)
    assert schema, f"build_obj require the '@type' property of the object to be populated! {dict_obj=}"
    assert schema in build_registry, f"{schema} was not registered to be built!"
    return build_registry[schema](dict_obj)
