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

from typing import Type, TypeVar, Iterable

from ..commons_new.register_recall import RegisterRecall

T = TypeVar("T")

# TODO investigating why register recall fails
# when encountering e.g. brainglobe register atlas elements
attribute_collection_iterator = RegisterRecall[list](cache=False)

# TODO push update rather than pull update for iter_collection
attribute_collection_iterator.on_new_registration = lambda *args, **kwargs: None


def iter_collection(_type: Type[T]) -> Iterable[T]:
    return [item for item in attribute_collection_iterator.iter(_type)]
