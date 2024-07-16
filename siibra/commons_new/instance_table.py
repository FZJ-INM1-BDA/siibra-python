from typing import Generic, Iterable, TypeVar, Dict, Iterator, Union, List, Callable

from .string import create_key
from ..commons_new.logger import logger

T = TypeVar("T")
V = TypeVar("V")


def default_comparison(a, b):
    return a == b


class BaseInstanceTable(Generic[T], Iterable):
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
        if matchfunc is None:
            matchfunc = default_comparison
        assert hasattr(
            matchfunc, "__call__"
        ), "matchfunc must implement __call__ method"

        if elements is None:
            elements: Dict[str, T] = {}

        assert isinstance(elements, dict), "Element must be of type dict"
        assert all(
            isinstance(k, str) for k in elements.keys()
        ), "All dictionary key must be type str"

        self._matchfunc: Callable[[T, V], bool] = matchfunc
        self._elements: Dict[str, T] = elements

    def __dir__(self):
        """List of all object keys in the registry"""
        return self.keys

    def __str__(self) -> str:
        if len(self) > 0:
            return f"{self.__class__.__name__}:\n - " + "\n - ".join(
                self._elements.keys()
            )
        return f"Empty {self.__class__.__name__}"

    def __repr__(self):
        return f"<{self.__class__.__name__} of {self[0].__class__}>"

    def __iter__(self) -> Iterator[T]:
        """Iterate over all objects in the registry"""
        return (w for w in self._elements.values())

    def __contains__(self, key: Union[str, T]) -> bool:
        """Test wether the given key or element is defined by the registry."""
        if key in self._elements:
            return True
        if key in self._elements.values():
            return True
        return False

    def __len__(self) -> int:
        """Return the number of elements in the registry"""
        return len(self._elements)

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
            raise IndexError(f"Provided {spec=} is larger than {len(self._elements)=}")

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

    def values(self):
        return list(self._elements.values())

    @property
    def keys(self):
        return [create_key(key) for key in self._elements.keys()]

    @property
    def names(self):
        return self.keys

    def __getattr__(self, index) -> T:
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


class InstanceTable(BaseInstanceTable[T]):
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
        super().__init__(matchfunc, elements)

    def add(self, key: str, value: T) -> None:
        """
        Add a key/value pair to the registry.

        Parameters
        ----------
            key (string): Unique name or key of the object
            value (object): The registered object
        """
        if key in self._elements:
            logger.error(
                f"Key {key} already in {__class__.__name__}, existing value will be replaced."
            )
        self._elements[key] = value

    def __sub__(self, obj) -> "InstanceTable[T]":
        """
        remove an object from the registry
        """
        if obj in self._elements.values():
            return InstanceTable[T](
                self._matchfunc, {k: v for k, v in self._elements.items() if v != obj}
            )
        else:
            return self


# TODO . accessor is really slow, investigate why
# I think python internally check each autocomplete value against __getattribute__
# this makes things very slow
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
