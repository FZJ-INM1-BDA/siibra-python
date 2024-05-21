from dataclasses import dataclass, field
from typing import List, Any
from ..commons import logger

SCHEMAS = {}


@dataclass
class Attribute:
    """Base clase for attributes."""

    schema: str = field(default="siibra/attr", init=False)

    # TODO performance implications? may have to set hash=False
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    # derived classes set their schema as class parameter
    def __init_subclass__(cls):
        assert (
            cls.schema != Attribute.schema
        ), "Subclassed attributes must have unique schemas"
        assert cls.schema not in SCHEMAS, f"{cls.schema} already registered."
        SCHEMAS[cls.schema] = cls

    @staticmethod
    def from_dict(json_dict: dict[str, Any]) -> List["Attribute"]:
        """Generating a list of attributes from a dictionary.
        TODO consider moving this to siibra.factory.factory and have a single build_object call"""
        
        att_type: str = json_dict.pop("@type")
        if att_type.startswith("x-"):
            return []
        Cls = SCHEMAS.get(att_type)
        if Cls is None:
            logger.warn(f"Cannot parse type {att_type}")
            return []

        return_attr: "Attribute" = Cls(
            **{key: json_dict[key] for key in json_dict if not key.startswith("x-")}
        )
        for key in json_dict:
            if key.startswith("x-"):
                return_attr.extra[key] = json_dict[key]
        return [return_attr]
