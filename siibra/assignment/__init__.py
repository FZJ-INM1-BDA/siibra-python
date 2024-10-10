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

from typing import TypeVar, List, Type, Generic, Iterable
from dataclasses import field
import pandas as pd

from .assignment import (
    match as collection_match,
    qualify as collection_qualify,
    preprocess_concept,
)
from ..factory.livequery import LiveQuery
from ..factory.configuration import iter_preconfigured
from ..attributes import AttributeCollection
from ..attributes.descriptions import ID, Name, Categorization
from ..cache import fn_call_cache

T = TypeVar("T", bound=AttributeCollection)


class SearchResult(Generic[T]):
    """
    Handles searching through both preconfigured, but also live-populated attribute collections.
    User of this class provide critieria as a list of AttributeCollections and the Type of
    attribute collection to search for. This class will yield all instances of requested type satisfying
    all of the criteria provided.

    For preconfigured instances, the "all" logic is enforced via `all(collection_match(...))`; for live
    query instances, it falls back to the individual implementation of the LiveQuery.generate method call.

    n.b. especially for preconfigured instances, order of the criteria can affect the search efficiency.
    opt for the criteria that is most likely to return false first (e.g. for feature, Modality)
    """

    def __init__(self, criteria=None, search_type: Type[T] = None):
        if search_type is None:
            raise RuntimeError(f"search_type must be defined!")
        self.search_type = search_type
        self.criteria = criteria or []

    criteria: List[AttributeCollection] = field(default_factory=list)
    search_type: Type[T] = None

    def find(self) -> List[T]:
        return SearchResult.cached_find(self.criteria, self.search_type)

    @staticmethod
    def _find_iter(criteria: List[AttributeCollection], search_type: Type[T]):

        from ..factory import iter_preconfigured

        for item in iter_preconfigured(search_type):
            if all(collection_match(cri, item) for cri in criteria):
                yield item

        for cls in LiveQuery.get_clss(search_type):
            inst = cls(criteria)
            yield from inst.generate()

    @staticmethod
    @fn_call_cache
    def cached_find(criteria: List[AttributeCollection], search_type: Type[T]):
        return list(SearchResult._find_iter(criteria, search_type))

    @staticmethod
    def str_search_criteria(input: str):
        id_attr = ID(value=input)
        name_attr = Name(value=input, shortform=input)
        query = AttributeCollection(attributes=[id_attr, name_attr])
        return [query]

    @staticmethod
    def build_summary_table(items: List[T]):
        """
        Returns a dataframe where user can inspect/evaluate how to further narrow down
        the search result. Similar to AttributeCollection.data_recipe_table
        """
        list_of_dict = [
            {
                "name": item.name,
                "modalities": item.modalities,
                # In case key is one of ID, name etc, prepend to avoid name collision
                "categorizations": item.categorizations,
                **{
                    f"category_{categorization.key}": categorization.value
                    for categorization in item._find(Categorization)
                },
                "ID": item.ID,
                "instance": item,
            }
            for item in items
        ]
        return pd.DataFrame(list_of_dict)

    @staticmethod
    def pick_instance(items: List[T], expr=None, index=None) -> T:
        """
        Allow user to apply what was learnt from get_summary_table and get a subset of the search.
        """
        df = SearchResult.build_summary_table(items)
        if expr is not None:
            return df.query(expr=expr).iloc[0]["instance"]
        if index is not None:
            return df.iloc[index]["instance"]
        raise NotImplementedError

    def get_summary_table(self):
        return self.build_summary_table(self.find())

    def get_instance(self, expr=None, index=None):
        """
        Allow user to apply what was learnt from get_summary_table and get a subset of the search.
        """
        return self.pick_instance(self.find(), expr=expr, index=index)
