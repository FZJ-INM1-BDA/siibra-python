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

from typing import TypeVar, List, Type

from .assignment import (
    match as collection_match,
    qualify as collection_qualify,
    preprocess_concept,
)
from ..factory.livequery import LiveQuery
from ..factory.iterator import iter_preconfigured_ac
from ..attributes import AttributeCollection
from ..attributes.descriptions import ID, Name

T = TypeVar("T", bound=AttributeCollection)


def find(criteria: List[AttributeCollection], find_type: Type[T]):
    return list(finditer(criteria, find_type))


def finditer(criteria: List[AttributeCollection], find_type: Type[T]):
    """Providing a list of AttributeCollection and Type. Yields instances of the given type.

    For preconfigured instances, will yield if and only if every instance of attribute_collection
    matches with the instance of _find_type.

    For LiveQuery instances, it is configured at runtime."""
    for item in iter_preconfigured_ac(find_type):
        if all(collection_match(cri, item) for cri in criteria):
            yield item
    for cls in LiveQuery.get_clss(find_type):
        inst = cls(criteria)
        yield from inst.generate()


def string_search(input: str, req_type: Type[T]) -> List[T]:
    id_attr = ID(value=input)
    name_attr = Name(value=input, shortform=input)
    query = AttributeCollection(attributes=[id_attr, name_attr])
    return find([query], req_type)
