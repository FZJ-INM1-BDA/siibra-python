# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field, replace
from typing import Tuple, Type, TypeVar, Iterable, Callable, List
import pandas as pd

from .attribute import Attribute
from ..attributes.descriptions import Url, Doi, TextDescription, Facet
from ..commons_new.iterable import assert_ooo

T = TypeVar("T")


MATRIX_INDEX_ENTITY_KEY = "x-siibra/matrix-index-entity/index"

def attr_of_general_interest(attr: Attribute):
    return MATRIX_INDEX_ENTITY_KEY in attr.extra


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
        from ..retrieval.doi_fetcher import get_citation

        citations = [
            Url(value=doi.value, text=get_citation(doi)) for doi in self._find(Doi)
        ]

        return [*self._find(Url), *citations]

    @property
    def description(self):
        return self._get(TextDescription).value

    @property
    def facets(self):
        df = pd.DataFrame(
            [
                {
                    "key": facet.key,
                    "value": facet.value
                } for facet in self._find(Facet)
            ]
        )
        return pd.concat([ df, *[attr.facets for attr in self.attributes]])
    
    @staticmethod
    def list_facets(attribute_collections: List["AttributeCollection"]):
        return pd.concat([ac.facets for ac in attribute_collections])

    @staticmethod
    def get_query_str(facet_dict=None, **kwargs):
        if facet_dict is not None:
            assert isinstance(facet_dict, dict), f"positional argument must be a dict"
            assert all((isinstance(key, str) and isinstance(value, str) for key, value in facet_dict.items())), f"Only string key value can be provided!"
        else:
            facet_dict = dict()

        return " | ".join([f"key=='{key}' & value=='{value}'" for key, value in {
            **facet_dict,
            **kwargs,
        }.items()])

    def filter_attributes_by_facets(self, facet_dict=None, **kwargs):
        query_str = AttributeCollection.get_query_str(facet_dict, **kwargs)

        return self.filter(
            lambda a: (
                attr_of_general_interest(a)
                or (len(a.facets) > 0 and len(a.facets.query(query_str)) > 0)
            )
        )

    @staticmethod
    def filter_facets(attribute_collections: List["AttributeCollection"], facet_dict=None, **kwargs):
        query_str = AttributeCollection.get_query_str(facet_dict, **kwargs)
        return [ac for ac in attribute_collections if len(ac.facets.query(query_str)) > 0]

    
    @staticmethod
    def find_facets(attribute_collections: List["AttributeCollection"]):
        return pd.concat([ac.facets for ac in attribute_collections])

    def relates_to(self, attribute_collection: "AttributeCollection"):
        """Yields attribute from self, attribute from target attribute_collection, and how they relate"""
        from ..assignment import collection_qualify
        yield from collection_qualify(self, attribute_collection)

    @property
    def ebrains_ids(self) -> Iterable[Tuple[str, str]]:
        """
        Yields all ebrains references as Iterable of Tuple, e.g.

        (
            ("openminds/ParcellationEntity", "foo"),
            ("minds/core/parcellationregion/v1.0.0", "bar"),
        )
        """
        from .descriptions import EbrainsRef

        for ebrainsref in self._finditer(EbrainsRef):
            for key, value in ebrainsref.ids.items():
                if isinstance(value, list):
                    for v in value:
                        yield key, v
                if isinstance(value, str):
                    yield key, value
