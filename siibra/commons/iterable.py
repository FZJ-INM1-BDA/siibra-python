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

from typing import TypeVar, Iterable, Callable
from ..exceptions import NonUniqueError

T = TypeVar("T")


def get_ooo(input: Iterable[T], filter_fn: Callable[[T], bool]) -> T:
    """Get One and Only One from an iterable

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
    result = list(filter(filter_fn, input))
    try:
        assert len(result) == 1
    except AssertionError as e:
        raise NonUniqueError from e
    return result[0]


def assert_ooo(input: Iterable[T]) -> T:
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
    assert len(listed_input) == 1, f"Expected one item, but got {len(input)}"
    return listed_input[0]
