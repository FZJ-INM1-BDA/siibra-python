
from typing import Iterator, List, Type, Dict, TypeVar, Generic
from abc import ABC, abstractmethod
from collections import defaultdict

from ...attributes import AttributeCollection, Attribute

T = TypeVar("T")
V = TypeVar("V", bound=AttributeCollection)
Y = TypeVar("Y", bound=Attribute)

class LiveQuery(ABC, Generic[T]):

    _ATTRIBUTE_COLLECTION_REGISTRY: Dict[Type[V], List[Type["LiveQuery[V]"]]] = defaultdict(list)

    def __init__(self, input: List[AttributeCollection]):
        self.input = input

    def __init_subclass__(cls, generates=Type[AttributeCollection]):
        cls._ATTRIBUTE_COLLECTION_REGISTRY[generates].append(cls)

    @classmethod
    def get_clss(cls, find_type: Type[V]):
        clss: List[Type[LiveQuery[V]]] = cls._ATTRIBUTE_COLLECTION_REGISTRY[find_type]
        return clss

    @abstractmethod
    def generate(self) -> Iterator[T]:
        raise NotImplementedError
    
    def find_attributes(self, type: Type[Y]):
        """Filters for the type indicated, returns List of List of attributes.
        Subclass is responsible for deciding what to do with this information
        """
        return [inp._find(type) for inp in self.input]

    def find_attribute_collections(self, type: Type[V]):
        """Returns a list of attribute_collection that is of the type asked"""
        return [inp for inp in self.input if isinstance(inp, type)]
