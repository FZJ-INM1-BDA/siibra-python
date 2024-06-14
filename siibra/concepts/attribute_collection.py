from dataclasses import dataclass, field, replace
from typing import Tuple, Type, TypeVar, Iterable, Callable

from .attribute import Attribute
from ..descriptions import Name, ID
from ..commons_new.iterable import assert_ooo

T = TypeVar("T")


@dataclass
class AttributeCollection:
    schema: str = "siibra/attribute_collection"
    attributes: Tuple[Attribute] = field(default_factory=list, repr=False)

    def _get(self, attr_type: Type[T]):
        return assert_ooo(self._find(attr_type))

    def _find(self, attr_type: Type[T]):
        return list(self._finditer(attr_type))

    def _finditer(self, attr_type: Type[T]) -> Iterable[T]:
        for attr in self.attributes:
            if isinstance(attr, attr_type):
                yield attr

    @property
    def name(self):
        return self._get(Name).value

    @property
    def id(self):
        return self._get(ID).value
    
    def filter(self, filter_fn: Callable[[Attribute], bool]):
        return replace(self, attributes=(attr for attr in self.attributes if filter_fn(attr)))

