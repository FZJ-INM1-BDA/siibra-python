from typing import Type, TypeVar, Iterable

from ..commons_new.register_recall import RegisterRecall

T = TypeVar("T")

# TODO investigating why register recall fails
# when encountering e.g. brainglobe register atlas elements
attribute_collection_iterator = RegisterRecall[[]](cache=False)


def iter_collection(_type: Type[T]) -> Iterable[T]:
    return [item for item in attribute_collection_iterator.iter(_type)]
