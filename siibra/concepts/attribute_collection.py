from dataclasses import dataclass, field
from typing import Tuple, Type

from .attribute import Attribute


@dataclass
class AttributeCollection:
    schema: str = "siibra/attrCln"
    attributes: Tuple[Attribute] = field(default_factory=list)

    def get(self, attr_type: Type[Attribute]):
        return tuple(att for att in self.attributes if isinstance(att, attr_type))
