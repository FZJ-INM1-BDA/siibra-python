from dataclasses import dataclass, field, replace
from typing import Tuple, Type, TypeVar, Iterable, Callable

from .attribute import Attribute
from ..descriptions import Url, Doi, TextDescription, Modality
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

    def filter(self, filter_fn: Callable[[Attribute], bool]):
        """
        Return a new `AttributeCollection` that is a copy of this one where the
        only the `Attributes` evaluating to True from `filter_fn` are collected.
        """
        return replace(
            self, attributes=tuple(attr for attr in self.attributes if filter_fn(attr))
        )

    @property
    def publications(self):
        from ..retrieval_new.doi_fetcher import get_citation

        citations = [
            Url(value=doi.value, text=get_citation(doi)) for doi in self._find(Doi)
        ]

        return [*self._find(Url), *citations]

    @property
    def description(self):
        return self._get(TextDescription).value

    @property
    def modalities(self):
        return [mod.value for mod in self._find(Modality)]
