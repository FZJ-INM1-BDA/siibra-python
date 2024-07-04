from typing import Dict, Callable, List, TypeVar
from functools import wraps

from ..concepts.attribute import Attribute
from ..concepts.attribute_collection import AttributeCollection
from ..concepts.feature import Feature
from ..descriptions import Name, SpeciesSpec, ID, RegionSpec
from ..commons import create_key
from ..commons_new.iterable import assert_ooo
from ..atlases import region, parcellation, space, parcellationmap, sparsemap


V = TypeVar("V")

build_registry: Dict[str, Callable[[Dict], V]] = {}


def parse_attributes(attribute_objs: List):
    return tuple(
        att
        for attribute_obj in attribute_objs
        for att in Attribute.from_dict(attribute_obj)
    )


def register_build_type(type_str: str):
    def outer(fn: Callable[[Dict], V]):

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
    attributes = parse_attributes(attribute_objs)
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
    attributes = parse_attributes(attribute_objs)

    name_attributes: List[Name] = list(
        filter(lambda a: isinstance(a, Name), attributes)
    )
    id_attributes: List[ID] = list(filter(lambda a: isinstance(a, ID), attributes))
    attr_species: List[SpeciesSpec] = list(
        filter(lambda a: isinstance(a, SpeciesSpec), attributes)
    )

    name_str = name_attributes[0].value

    attributes += (RegionSpec(value=name_str, parcellation_id=parc_id.value),)
    if len(id_attributes) == 0:
        attributes += (ID(value=f"{parc_id.value}_{create_key(name_str)}"),)

    if len(attr_species) == 0:
        attributes += (species,)
    else:
        assert all(
            existing_spec == species for existing_spec in attr_species
        ), f"attribute species {attr_species} does not equal to passed specices {species}"

    return region.Region(
        attributes=attributes,
        children=tuple(
            map(
                lambda r: build_region(r, parc_id, species),
                dict_obj.get("children", []),
            )
        ),
    )


@register_build_type(parcellation.Parcellation.schema)
def build_parcellation(dict_obj: dict):
    dict_obj.pop("@type", None)
    attribute_objs = dict_obj.pop("attributes", [])
    attributes = parse_attributes(attribute_objs)

    id_attribute: ID = assert_ooo(filter(lambda a: isinstance(a, ID), attributes))
    species: SpeciesSpec = assert_ooo(
        filter(lambda a: isinstance(a, SpeciesSpec), attributes)
    )
    name_attribute: Name = assert_ooo(
        [attr for attr in attributes if isinstance(attr, Name)]
    )

    attributes += (
        RegionSpec(parcellation_id=id_attribute.value, value=name_attribute.value),
    )

    return parcellation.Parcellation(
        attributes=attributes,
        children=tuple(
            map(
                lambda r: build_region(r, id_attribute, species),
                dict_obj.get("regions", []),
            )
        ),
    )


@register_build_type(space.Space.schema)
def build_space(dict_obj):
    dict_obj.pop("@type", None)
    attribute_objs = dict_obj.pop("attributes", [])
    attributes = parse_attributes(attribute_objs)
    return space.Space(attributes=attributes)


@register_build_type(parcellationmap.Map.schema)
def build_map(dict_obj):
    dict_obj.pop("@type", None)
    if dict_obj.pop("sparsemap", False):
        MapType = sparsemap.SparseMap
    else:
        MapType = parcellationmap.Map

    region_attributes = {}
    for regionname, attrs in dict_obj.pop("region_attributes", {}).items():
        region_attributes[regionname] = AttributeCollection(
            attributes=parse_attributes(attrs)
        )

    attribute_objs = dict_obj.pop("attributes", [])
    attributes = parse_attributes(attribute_objs)

    return MapType(attributes=attributes, _region_attributes=region_attributes, **dict_obj)


def build_object(dict_obj: Dict):
    schema = dict_obj.get("@type", None)  # TODO: consider popping instead
    assert (
        schema
    ), f"build_obj require the '@type' property of the object to be populated! {dict_obj=}"
    assert schema in build_registry, f"{schema} was not registered to be built!"
    return build_registry[schema](dict_obj)
