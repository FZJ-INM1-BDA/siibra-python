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

from typing import TypeVar, Iterable, Callable, List
from ..exceptions import NonUniqueError

T = TypeVar("T")


def assert_ooo(input: Iterable[T], error_msg_cb: Callable[[List[T]], str] = None) -> T:
    """Assert One and Only One from an iterable, and return the only element.

    Parameters
    ----------
    input: Iterable[T]

    Returns
    -------
    T

    Raises
    ------
    NonUniqueError
    """
    listed_input = list(input)
    try:
        assert len(listed_input) == 1
    except AssertionError as e:
        msg = (
            error_msg_cb(listed_input)
            if error_msg_cb
            else f"Expected one item, but got {len(input)}"
        )
        raise NonUniqueError(msg) from e
    return listed_input[0]


def flatmap(list_of_list: List[List[T]]):
    return [item for _list in list_of_list for item in _list]
