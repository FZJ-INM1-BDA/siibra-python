from dataclasses import dataclass, field
from typing import List, Type

from .attribute import Attribute

@dataclass
class AttributeCollection:
    schema: str = "siibra/attrCln"
    attributes: List[Attribute] = field(default_factory=list)

    def get(self, attr_type: Type[Attribute]):
        return [attr for attr in self.attributes if isinstance(attr, attr_type)]

