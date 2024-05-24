from dataclasses import dataclass, field
from typing import Tuple, Type, TypeVar, Iterable

from .attribute import Attribute

T = TypeVar("T")


@dataclass
class AttributeCollection:
    schema: str = "siibra/attribute_collection"
    attributes: Tuple[Attribute] = field(default_factory=list, repr=False)

    def get(self, attr_type: Type[T]):
        return list(self.getiter(attr_type))

    def getiter(self, attr_type: Type[T]) -> Iterable[T]:
        for attr in self.attributes:
            if isinstance(attr, attr_type):
                yield attr
