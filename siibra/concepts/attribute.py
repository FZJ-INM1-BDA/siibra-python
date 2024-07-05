from dataclasses import dataclass, field
from typing import List, Any
from ..commons import logger

SCHEMAS = {}


def key_is_extra(key: str):
    return key.startswith("x-") or key.startswith("aggregate-by/")


@dataclass
class Attribute:
    """Base clase for attributes."""

    schema: str = field(default="siibra/attr", init=False, repr=False)

    # TODO performance implications? may have to set hash=False
    extra: dict[str, Any] = field(default_factory=dict, repr=False, hash=False)

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
        TODO consider moving this to siibra.factory.factory and have a single build_object call
        """

        att_type: str = json_dict.pop("@type")
        if att_type.startswith("x-"):
            return []
        Cls = SCHEMAS.get(att_type)
        if Cls is None:
            logger.warning(f"Cannot parse type {att_type}")
            return []

        return_attr: "Attribute" = Cls(
            **{key: json_dict[key] for key in json_dict if not key_is_extra(key)}
        )
        for key in json_dict:
            if key_is_extra(key):
                return_attr.extra[key] = json_dict[key]
        return [return_attr]

    @property
    def aggregate_by(self):
        from ..descriptions import AggregateBy

        # TODO use str.removeprefix when py3.9 is the lowest python version supported
        return [
            AggregateBy(key=key.replace("aggregate-by/", ""), value=self.extra[key])
            for key in self.extra
            if key.startswith("aggregate-by/")
        ]
