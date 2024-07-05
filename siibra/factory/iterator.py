from typing import Type, TypeVar, Iterable

from ..commons_new.register_recall import RegisterRecall

T = TypeVar("T")
attribute_collection_iterator = RegisterRecall[[]]()


def iter_collection(_type: Type[T]) -> Iterable[T]:
    return [item for item in attribute_collection_iterator.iter(_type)]
