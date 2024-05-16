from typing import Type, Callable, Dict, Iterable, List, TypeVar
from functools import wraps
from collections import defaultdict

from .attribute_match import match as attribute_match

from ..concepts.attribute_collection import AttributeCollection
from ..concepts.feature import Feature
from ..descriptions import Modality

T = Callable[[AttributeCollection], Iterable[AttributeCollection]]

collection_gen: Dict[Type[AttributeCollection], List[T]] = defaultdict(list)

def register_collection_generator(type_of_col: Type[AttributeCollection]):
    def outer(fn: T):
        collection_gen[type_of_col].append(fn)
        @wraps(fn)
        def inner(*args, **kwargs):
            return fn(*args, **kwargs)
        return inner
    return outer


V = TypeVar("V")


def get(input: AttributeCollection, req_type: Type[V]) -> Iterable[V]:
    for fn in collection_gen[req_type]:
        yield from fn(input)

def match(col_a: AttributeCollection, col_b: AttributeCollection) -> bool:
    return any(
        attribute_match(attra, attrb)
        for attra in col_a.attributes
        for attrb in col_b.attributes
    )
