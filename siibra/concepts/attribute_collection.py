from dataclasses import dataclass, field
from typing import Tuple, Type, TypeVar

from .attribute import Attribute

T = TypeVar('T')


@dataclass
class AttributeCollection:
    schema: str = "siibra/attribute_collection"
    attributes: Tuple[Attribute] = field(default_factory=list, repr=False)

    def get(self, attr_type: Type[Attribute]):
        return tuple(att for att in self.attributes if isinstance(att, attr_type))
