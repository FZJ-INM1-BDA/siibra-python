from dataclasses import dataclass, field
from typing import List, Any
from ...commons import logger

SCHEMAS = {}


@dataclass(kw_only=True)
class Attribute:
    """ Base clase for attributes. """
    schema: str = field(default="siibra/attr", init=False)
    extra: dict[str, Any] = field(default_factory=dict)

    # derived classes set their schema as class parameter
    def __init_subclass__(cls):
        assert cls.schema != Attribute.schema, f"Subclassed attributes must have unique schemas"
        assert cls.schema not in SCHEMAS, f"{cls.schema} already registered."
        SCHEMAS[cls.schema] = cls

    def matches(self, *args, **kwargs):
        return False
    
    @staticmethod
    def from_dict(json_dict: dict[str, Any]) -> List['Attribute']:
        att_type: str = json_dict.pop("@type")
        if att_type.startswith("x-"):
            return []
        Cls = SCHEMAS.get(att_type)
        if Cls is None:
            logger.warn(f"Cannot parse type {att_type}")
            return []

        return_attr: 'Attribute' = Cls(**{
            key: json_dict[key]
            for key in json_dict
            if not key.startswith("x-")
        })
        for key in json_dict:
            if key.startswith("x-"):
                return_attr.extra[key] = json_dict[key]
        return [return_attr]

