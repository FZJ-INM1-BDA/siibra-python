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

from typing import (
    Generic,
    Iterable,
    TypeVar,
    Dict,
    Iterator,
    Union,
    List,
    Callable
)

from .string import create_key
from ..commons_new.logger import logger

T = TypeVar("T")
V = TypeVar("V")


def default_comparison(a, b):
    return a == b


class TabCompleteCollection(Generic[T], Iterable):
    def __init__(self, elements: Union[Dict[str, T], None] = None) -> None:
        if elements is None:
            elements: Dict[str, T] = {}

        assert isinstance(elements, dict), "Element must be of type dict"
        assert all(
            isinstance(k, str) for k in elements.keys()
        ), "All dictionary key must be type str"
        self._elements: Dict[str, T] = elements

    def values(self):
        return list(self._elements.values())

    @property
    def keys(self):
        return [create_key(key) for key in self._elements.keys()]

    def __contains__(self, key: Union[str, T]) -> bool:
        """Test wether the given key or element is defined by the registry."""
        if key in self._elements:
            return True
        if key in self._elements.values():
            return True
        return False

    def __dir__(self):
        """List of all object keys in the registry"""
        return self.keys

    def __iter__(self) -> Iterator[T]:
        """Iterate over all objects in the registry"""
        return (w for w in self._elements.values())

    def __len__(self) -> int:
        """Return the number of elements in the registry"""
        return len(self._elements)

    def __getattr__(self, index: str) -> T:
        """Access elements by using their keys as attributes.
        Keys are auto-generated from the provided names to be uppercase,
        with words delimited using underscores.
        """
        _elements = self._elements
        keys = self.keys
        if index in _elements:
            return _elements[index]
        if index in keys:
            matched_item = [
                item for key, item in _elements.items() if create_key(key) == index
            ]
            assert len(matched_item) == 1
            return matched_item[0]
        hint = ""
        if isinstance(index, str):
            import difflib

            closest = difflib.get_close_matches(index, list(_elements.keys()), n=3)
            if len(closest) > 0:
                hint = f"Did you mean {' or '.join(closest)}?"
        raise AttributeError(f"Term '{index}' not in {__class__.__name__}. " + hint)


class BkwdCompatInstanceTable(TabCompleteCollection[T]):
    """
    Backwards compatible instance table for space, parcellation and map.
    siibra v2.0 introduced generic string matching. Rather than keep two
    separate maching mechanism (instance table & attribute based matching),
    instance table matching will be retired. To maintain some backwards-
    compatibility, this class act exposes the API similar to that of instance
    table. Underneath, it uses the new attribute based matching.
    """

    def __init__(
        self, getitem: Callable[[str], T], elements: Union[Dict[str, T], None] = None
    ) -> None:
        super().__init__(elements)
        self._getitem = getitem

    def __str__(self) -> str:
        if len(self) > 0:
            return f"{self.__class__.__name__}:\n - " + "\n - ".join(
                self._elements.keys()
            )
        return f"Empty {self.__class__.__name__}"

    def __getitem__(self, key: str):
        return self._getitem(key)

    def get(self, spec: str):
        return self._getitem(spec)


class BaseInstanceTable(TabCompleteCollection[T]):
    """
    Lookup table for instances of a given class by name/id.
    Provide attribute-access and iteration to a set of named elements,
    given by a dictionary with keys of 'str' type.
    """

    def __init__(
        self, matchfunc: Callable[[T, V], bool] = None, elements: Dict[str, T] = None
    ):
        """
        Build an object lookup table from a dictionary with string keys, for easy
        attribute-like access, name autocompletion, and iteration.
        Matchfunc can be provided to enable inexact matching inside the index operator.
        It is a binary function, taking as first argument a value of the dictionary
        (ie. an object that you put into this glossary), and as second argument
        the index/specification that should match one of the objects, and returning a boolean.
        """
        super().__init__(elements)
        if matchfunc is None:
            matchfunc = default_comparison
        assert hasattr(
            matchfunc, "__call__"
        ), "matchfunc must implement __call__ method"
        self._matchfunc: Callable[[T, V], bool] = matchfunc

    def __str__(self) -> str:
        if len(self) > 0:
            return f"{self.__class__.__name__}:\n - " + "\n - ".join(
                self._elements.keys()
            )
        return f"Empty {self.__class__.__name__}"

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self[0].__class__}>"

    def __getitem__(self, spec) -> T:
        return self.get(spec)

    def get(self, spec) -> T:
        """Give access to objects in the registry by sequential index,
        exact key, or keyword matching. If the keywords match multiple objects,
        the first in sorted order is returned. If the specification does not match,
        a RuntimeError is raised.

        Parameters
        ----------
        spec: int, str
            Index or string specification of an object

        Returns
        -------
            Matched object
        """
        if spec is None:
            return None
        elif spec == "":
            raise IndexError(f"{__class__.__name__} indexed with empty string")
        matches = self.find(spec)
        if len(matches) == 0:
            raise IndexError(
                f"{__class__.__name__} has no entry matching the specification '{spec}'."
                f"Possible values are:\n" + str(self)
            )
        elif len(matches) == 1:
            return matches[0]
        else:
            try:
                S = sorted(matches, reverse=True)
            except TypeError:
                # not all object types support sorting, accept this
                S = matches
            largest = S[0]
            logger.info(
                f"Multiple elements matched the specification '{spec}' - the first in order was chosen: {largest}"
            )
            return largest

    def provides(self, spec) -> bool:
        """
        Returns True if an element that matches the given specification can be found
        (using find(), thus going beyond the matching of names only as __contains__ does)
        """
        matches = self.find(spec)
        return len(matches) > 0

    def find(self, spec: Union[str, V]) -> List[T]:
        """
        Return a list of items matching the given specification,
        which could be either the name or a specification that
        works with the matchfunc of the Glossary.

        Raises
        ------
        IndexError: if spec is an empty str
        """
        if spec in self._elements:
            return [self._elements[spec]]

        if isinstance(spec, int):
            if spec < len(self._elements):
                return [list(self._elements.values())[spec]]
            raise IndexError(
                f"Provided spec={spec!r} is larger than {len(self._elements)}"
            )

        if isinstance(spec, str):
            # string matching on keys
            if spec == "":
                raise IndexError(f"{__class__.__name__} indexed with empty string")
            return [
                self._elements[k]
                for k in self._elements.keys()
                if all(w.lower() in k.lower() for w in spec.split())
            ]

        return [v for v in self._elements.values() if self._matchfunc(v, spec)]

    @property
    def names(self):
        return self.keys


class JitInstanceTable(BaseInstanceTable[T]):
    def __init__(
        self,
        matchfunc: Callable[[T, V], bool] = None,
        getitem: Callable[[], Dict[str, T]] = None,
    ):
        super().__init__(matchfunc)
        if getitem is None:
            raise RuntimeError(
                "JitInstanceTable initialization must have getitem defined!"
            )
        assert getitem
        self.getitem = getitem

    @property
    def _elements(self):
        try:
            return self.getitem()
        except Exception as e:
            print(e)
            return {}

    @_elements.setter
    def _elements(self, value):
        pass
