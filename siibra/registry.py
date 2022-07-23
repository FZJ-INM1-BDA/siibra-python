# Copyright 2018-2021
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

from . import __version__
from .commons import logger, QUIET
from .retrieval.repositories import GitlabConnector, RepositoryConnector
from .retrieval.exceptions import NoSiibraConfigMirrorsAvailableException, TagNotFoundException

import os
from typing import Any, Generic, Iterable, Iterator, List, TypeVar


# Until openminds is fully supported, we get configurations of siibra concepts from gitlab.
GITLAB_PROJECT_TAG = os.getenv(
    "SIIBRA_CONFIG_GITLAB_PROJECT_TAG", "siibra-{}".format(__version__)
)
USE_DEFAULT_PROJECT_TAG = "SIIBRA_CONFIG_GITLAB_PROJECT_TAG" not in os.environ


T = TypeVar('T')


class TypedRegistry(Generic[T], Iterable):
    """
    Provide attribute-access and iteration to a set of named elements,
    given by a dictionary with keys of 'str' type.
    """

    def __init__(self, matchfunc=lambda a, b: a == b, elements=None):
        """
        Build a glossary from a dictionary with string keys, for easy
        attribute-like access, name autocompletion, and iteration.
        Matchfunc can be provided to enable inexact matching inside the index operator.
        It is a binary function, taking as first argument a value of the dictionary
        (ie. an object that you put into this glossary), and as second argument
        the index/specification that should match one of the objects, and returning a boolean.
        """

        assert hasattr(matchfunc, "__call__")
        if elements is None:
            self._elements = {}
        else:
            assert isinstance(elements, dict)
            assert all(isinstance(k, str) for k in elements.keys())
            self._elements = elements
        self._matchfunc = matchfunc

    def add(self, key: str, value: T) -> None:
        """Add a key/value pair to the registry.

        Args:
            key (string): Unique name or key of the object
            value (object): The registered object
        """
        assert isinstance(key, str)
        if key in self._elements:
            logger.warning(
                f"Key {key} already in {__class__.__name__}, existing value will be replaced."
            )
        self._elements[key] = value

    def __dir__(self) -> Iterable[str]:
        """List of all object keys in the registry"""
        return self._elements.keys()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: " + ",".join(self._elements.keys())

    def __iter__(self) -> Iterator[T]:
        """Iterate over all objects in the registry"""
        return (w for w in self._elements.values())

    def __contains__(self, key) -> bool:
        """Test wether the given key is defined by the registry."""
        return (
            key in self._elements
        )  # or any([self._matchfunc(v,spec) for v in self._elements.values()])

    def __len__(self) -> int:
        """Return the number of elements in the registry"""
        return len(self._elements)

    def __getitem__(self, spec) -> T:
        """Give access to objects in the registry by sequential index,
        exact key, or keyword matching. If the keywords match multiple objects,
        the first in sorted order is returned. If the specification does not match,
        a RuntimeError is raised.

        Args:
            spec [int or str]: Index or string specification of an object

        Returns:
            Matched object
        """
        if spec is None:
            raise RuntimeError(f"{__class__.__name__} indexed with None")
        matches = self.find(spec)
        if len(matches) == 0:
            print(str(self))
            raise IndexError(
                f"{__class__.__name__} has no entry matching the specification '{spec}'.\n"
                f"Possible values are:\n - " + "\n - ".join(self._elements.keys())
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

    def __sub__(self, obj) -> 'TypedRegistry[T]':
        """
        remove an object from the registry
        """
        if obj in self._elements.values():
            return TypedRegistry[T](
                self._matchfunc, {k: v for k, v in self._elements.items() if v != obj}
            )
        else:
            return self

    def provides(self, spec) -> bool:
        """
        Returns True if an element that matches the given specification can be found
        (using find(), thus going beyond the matching of names only as __contains__ does)
        """
        matches = self.find(spec)
        return len(matches) > 0

    def find(self, spec) -> List[T]:
        """
        Return a list of items matching the given specification,
        which could be either the name or a specification that
        works with the matchfunc of the Glossary.
        """
        if isinstance(spec, str) and (spec in self._elements):
            return [self._elements[spec]]
        elif isinstance(spec, int) and (spec < len(self._elements)):
            return [list(self._elements.values())[spec]]
        else:
            # string matching on values
            matches = [v for v in self._elements.values() if self._matchfunc(v, spec)]
            if len(matches) == 0:
                # string matching on keys
                matches = [
                    self._elements[k]
                    for k in self._elements.keys()
                    if all(w.lower() in k.lower() for w in spec.split())
                ]
            return matches

    def __getattr__(self, index) -> T:
        """Access elements by using their keys as attributes.
        Keys are auto-generated from the provided names to be uppercase,
        with words delimited using underscores.
        """
        if index in self._elements:
            return self._elements[index]
        else:
            hint = ""
            if isinstance(index, str):
                import difflib

                closest = difflib.get_close_matches(
                    index, list(self._elements.keys()), n=3
                )
                if len(closest) > 0:
                    hint = f"Did you mean {' or '.join(closest)}?"
            raise AttributeError(f"Term '{index}' not in {__class__.__name__}. " + hint)


class Registry(TypedRegistry[Any]):
    pass


class PreconfiguredObjects:
    """
    Registry of preconfigured/bootstrapped objects of differenct siibra classes.
    For each class which is decorated by @Preconfigure (see below),
    a registry of predefined objects will be created.
    Only when first requested, the registry will be populated with objects
    of that class, bootstrapped from specifications in a particular subfolder
    of the siibra configuration. The subfolder is determined by the 
    class decorator.

    Predefined objects are shared between all instances of this class -
    no duplicate objects will be created when creating multiple instances of this class.
    We could implement a strict singleton in the future.

    For example, 
        @Preconfigure("atlases")
        class Atlas...

    will make inform the PreconfiguredObjects class to provide a registry of predefined "Atlas" objects, 
    and (when first requested) bootstrap objects from the "atlases" subfolder of the siibra configuration.

    Configurations are by default fetched from the siibra-configurations repository maintained at Forschungszentrum Jülich.
    """

    _folders = {}
    _objects = {}

    _CONNECTORS = [
        GitlabConnector(
            "https://jugit.fz-juelich.de",
            3484,
            GITLAB_PROJECT_TAG,
            skip_branchtest=USE_DEFAULT_PROJECT_TAG,
        ),
        GitlabConnector(
            "https://gitlab.ebrains.eu",
            93,
            GITLAB_PROJECT_TAG,
            skip_branchtest=USE_DEFAULT_PROJECT_TAG,
        ),
    ]

    @classmethod
    def add_configuration(cls, conn: RepositoryConnector):
        cls._CONNECTORS.insert(0, conn)

    @classmethod
    def use_configuration(cls, conn: RepositoryConnector):
        cls._CONNECTORS = [conn]

    @classmethod
    def bootstrap(cls, registered_cls):

        # at this point we have to know the bootstrap folder of the given class
        assert registered_cls in cls._folders

        # at this point we should not have a registry for this class yet, and will create it.
        assert registered_cls not in cls._objects
        cls._objects[registered_cls] = TypedRegistry[registered_cls](matchfunc=registered_cls.match_spec)

        # fill the registry with new bootstrapped object instances
        for connector in cls._CONNECTORS:
            try:
                loaders = connector.get_loaders(
                    cls._folders[registered_cls],
                    ".json",
                    progress=f"Bootstrap: {registered_cls.__name__:15.15}",
                )
                break
            except Exception as e:
                print(str(e))
                logger.error(f"Cannot connect to configuration server {str(connector)}")
                *_, last = cls._CONNECTORS
                if connector is last:
                    raise NoSiibraConfigMirrorsAvailableException(
                        "Tried all mirrors, none available."
                    )

        else:
            # we get here only if the loop is not broken
            raise TagNotFoundException(
                f"Cannot bootstrap '{registered_cls.__name__}' objects: No configuration data found for '{GITLAB_PROJECT_TAG}'."
            )

        with QUIET:
            for fname, loader in loaders:
                obj = registered_cls._from_json(loader.data)
                if isinstance(obj, registered_cls):
                    cls._objects[registered_cls].add(obj.key, obj)
                else:
                    raise RuntimeError(
                        f"Could not generate object of type {registered_cls} from configuration {fname} - construction provided type {obj.__class__}"
                    )

    def __getitem__(self, cls):
        """
        Access predefined object registries by class, e.g.
        REGISTRY[Atlas]
        """
        assert cls in self._folders
        if cls not in self._objects:
            self.bootstrap(cls)
        return self._objects[cls]

    def __getattr__(self, attr):
        """
        Access pr3edefined object registries by attribute, e.g.
        REGISTRY.Atlas
        """
        if attr in self._folders:
            return self.__getitem__(attr)
        else:
            raise AttributeError


class Preconfigure:
    """
    Decorator for preconfiguring instances of siibra classes from siibra configuration files.

    Requires to provide the configuration subfolder which contains json files for bootstrapping objects of that class.

    For example, 
        @Preconfigure("atlases")
        class Atlas...

    will make inform the PreconfiguredObjects class to provide a registry of predefined "Atlas" objects, 
    and (when first requested) bootstrap objects from the "atlases" subfolder of the siibra configuration.
    """

    def __init__(self, folder):
        self.folder = folder

    def __call__(self, cls):
        if not all([hasattr(cls, "_from_json"), callable(cls._from_json)]):
            raise RuntimeError(
                f"Class '{cls.__name__}' needs to implement '_from_json()' "
                "in order to use the @preconfigure decorator."
            )
        PreconfiguredObjects._folders[cls] = self.folder
        return cls


REGISTRY = PreconfiguredObjects()
