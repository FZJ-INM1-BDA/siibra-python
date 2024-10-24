from typing import Iterator, List, Type, Dict, TypeVar, Generic
from abc import ABC, abstractmethod
from collections import defaultdict

from ...attributes import AttributeCollection, Attribute

T = TypeVar("T")
V = TypeVar("V", bound=AttributeCollection)
Y = TypeVar("Y", bound=Attribute)


class LiveQuery(ABC, Generic[T]):
    """
    Base class where AttributeCollection can be generated on-the-fly.

    Deriving class is as follows:

    ```
    class DerviedLiveQuery(LiveQuery, generates=Feature): pass
    ```

    """

    _ATTRIBUTE_COLLECTION_REGISTRY: Dict[Type[V], List[Type["LiveQuery[V]"]]] = (
        defaultdict(list)
    )

    def __init__(self, input: List[AttributeCollection]):
        self.input = input

    def __init_subclass__(cls, generates=Type[AttributeCollection]):
        cls._ATTRIBUTE_COLLECTION_REGISTRY[generates].append(cls)

    @classmethod
    def iter_livequery_instances(
        cls, find_type: Type[V], criteria: List[AttributeCollection]
    ):
        clss: List[Type[LiveQuery[V]]] = cls._ATTRIBUTE_COLLECTION_REGISTRY[find_type]
        for cls in clss:
            for criterion in criteria:
                if not cls.needs(criterion):
                    continue
            yield cls(criteria)

    @abstractmethod
    def generate(self) -> Iterator[T]:
        """
        Derived classes must override this method.

        siibra will call `generate()` to retrieve instances of type T, accordingly to the specifications (instance.inputs)
        This method can be slow/heavy.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def needs(cls, ac: AttributeCollection) -> bool:
        """
        Derived classes should override this classmethod.

        siibra will use the implementation to filter LiveQuery classes that does **not** match input criteria.

        This method should be quick (e.g. **not** network bound, does **not** require heavy computation).
        Dervied class should over promise (i.e. return True), as they can still opt to return empty Iterable when
        `generate()` is called.
        """
        return False

    def find_attributes(self, type: Type[Y]):
        """Filters for the type indicated, returns List of List of attributes.
        Subclass is responsible for deciding what to do with this information
        """
        return [inp._find(type) for inp in self.input]

    def find_attribute_collections(self, type: Type[V]):
        """Returns a list of attribute_collection that is of the type asked"""
        return [inp for inp in self.input if isinstance(inp, type)]
