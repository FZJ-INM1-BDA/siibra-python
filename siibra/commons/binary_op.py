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

from typing import TypeVar, Generic, Callable, Type, Dict, Tuple

T = TypeVar("T")
V = TypeVar("V")


class BinaryOpRegistry(Generic[T, V]):
    """
    Create a container for methods that takes two objects.

    Note
    ----
    Assumes the operation is commutative. For an example of handling
    non-commutative see `attributres.locations.ops.union` module.
    """

    def __init__(self):
        self._store_dict: Dict[
            Tuple[Type[T], Type[T]], Tuple[Callable[[T, T], V], bool]
        ] = {}
        self._registered_types = set()

    def register(self, a: Type[T], b: Type[T]):
        self._registered_types.add(a)
        self._registered_types.add(b)

        def outer(fn):
            forward_key = a, b
            backward_key = b, a

            assert forward_key not in self._store_dict, f"{forward_key} already exist"
            assert backward_key not in self._store_dict, f"{backward_key} already exist"

            self._store_dict[backward_key] = fn, True
            self._store_dict[forward_key] = fn, False

            return fn

        return outer

    def get(self, a: T, b: T):
        typea = type(a)
        typeb = type(b)
        key = typea, typeb
        return self._store_dict.get(key)

    def is_registered(self, t: Type):
        return t in self._registered_types
