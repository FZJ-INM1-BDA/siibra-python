from dataclasses import dataclass, field
from typing import List, Type, TypeVar

from .attribute import Attribute

T = TypeVar('T')

@dataclass
class AttributeCollection:
    schema: str = "siibra/attrCln"
    attributes: List[Attribute] = field(default_factory=list, repr=False)

    def get(self, attr_type: Type[T]) -> List[T]:
        return [attr for attr in self.attributes if isinstance(attr, attr_type)]

