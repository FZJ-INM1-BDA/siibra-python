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
from dataclasses import dataclass, field

from .assignment import (
    match as collection_match,
    qualify as collection_qualify,
    preprocess_concept,
)
from ..factory.livequery import LiveQuery
from ..factory.configuration import iter_preconfigured
from ..attributes import AttributeCollection
from ..attributes.descriptions import ID, Name

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

    def find_iter(self) -> Iterable[T]:
        from ..factory import iter_preconfigured

        for item in iter_preconfigured(self.search_type):
            if all(collection_match(cri, item) for cri in self.criteria):
                yield item

        for cls in LiveQuery.get_clss(self.search_type):
            inst = cls(self.criteria)
            yield from inst.generate()

    def find(self):
        self.search_type
        return list(self.find_iter())

    @staticmethod
    def str_search_criteria(input: str):
        id_attr = ID(value=input)
        name_attr = Name(value=input, shortform=input)
        query = AttributeCollection(attributes=[id_attr, name_attr])
        return [query]
