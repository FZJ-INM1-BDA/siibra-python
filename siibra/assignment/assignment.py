from typing import Type, Callable, Dict, Iterable, List, TypeVar
from functools import wraps
from collections import defaultdict
from itertools import product

from .attribute_match import match as attribute_match

from ..commons import logger
from ..concepts.attribute_collection import AttributeCollection
from ..concepts.feature import Feature
from ..descriptions import Modality
from ..exceptions import InvalidAttrCompException, UnregisteredAttrCompException

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
        try:
            yield from fn(input)
        except UnregisteredAttrCompException:
            continue

def match(col_a: AttributeCollection, col_b: AttributeCollection) -> bool:
    
    for attra, attrb in product(col_a.attributes, col_b.attributes):
        try:
            if attribute_match(attra, attrb):
                return True
        except UnregisteredAttrCompException as e:
            continue
        except InvalidAttrCompException as e:
            logger.warn(f"match exception {e}")
            return False
        
    raise UnregisteredAttrCompException
