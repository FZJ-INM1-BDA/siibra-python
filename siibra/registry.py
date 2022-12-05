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
from .commons import logger
from .retrieval.repositories import GitlabConnector, RepositoryConnector
from .retrieval.exceptions import NoSiibraConfigMirrorsAvailableException
from .config import SIIBRA_USE_CONFIGURATION

from typing import Generic, Iterable, Iterator, List, TypeVar, Union, Dict, Type
from collections import defaultdict
from requests.exceptions import ConnectionError

from abc import ABC, abstractmethod


T = TypeVar("T")


class InstanceTable(Generic[T], Iterable):
    """
    Lookup table for instances of a given class by name/id.
    Provide attribute-access and iteration to a set of named elements,
    given by a dictionary with keys of 'str' type.
    """

    def __init__(self, matchfunc=lambda a, b: a == b, elements=None):
        """
        Build an object lookup table from a dictionary with string keys, for easy
        attribute-like access, name autocompletion, and iteration.
        Matchfunc can be provided to enable inexact matching inside the index operator.
        It is a binary function, taking as first argument a value of the dictionary
        (ie. an object that you put into this glossary), and as second argument
        the index/specification that should match one of the objects, and returning a boolean.
        """

        assert hasattr(matchfunc, "__call__")
        if elements is None:
            self._elements: Dict[str, T] = {}
        else:
            assert isinstance(elements, dict)
            assert all(isinstance(k, str) for k in elements.keys())
            self._elements: Dict[str, T] = elements
        self._matchfunc = matchfunc

    def add(self, key: str, value: T) -> None:
        """Add a key/value pair to the registry.

        Args:
            key (string): Unique name or key of the object
            value (object): The registered object
        """
        if key in self._elements:
            logger.error(
                f"Key {key} already in {__class__.__name__}, existing value will be replaced."
            )
        self._elements[key] = value

    def __dir__(self) -> Iterable[str]:
        """List of all object keys in the registry"""
        return self._elements.keys()

    def __str__(self) -> str:
        if len(self) > 0:
            return f"{self.__class__.__name__}:\n - " + "\n - ".join(self._elements.keys())
        else:
            return f"Empty {self.__class__.__name__}"

    def __iter__(self) -> Iterator[T]:
        """Iterate over all objects in the registry"""
        return (w for w in self._elements.values())

    def __contains__(self, key: Union[str, T]) -> bool:
        """Test wether the given key is defined by the registry."""
        if isinstance(key, str):
            return key in self._elements
        return key in [item for _, item in self._elements.values()]


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
            raise IndexError(f"{__class__.__name__} indexed with None")
        elif spec == "":
            raise IndexError(f"{__class__.__name__} indexed with empty string")
        matches = self.find(spec)
        if len(matches) == 0:
            print(str(self))
            raise IndexError(
                f"{__class__.__name__} has no entry matching the specification '{spec}'.\n"
                f"Possible values are: " + ", ".join(self._elements.keys())
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


class Query(ABC):

    # set of mandatory query argument names
    _query_args = []

    def __init__(self, **kwargs):
        parstr = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        if parstr:
            parstr = "with parameters " + parstr
        logger.info(f"Initializing query for {self._FEATURETYPE.__name__} features {parstr}")
        if not all(p in kwargs for p in self._query_args):
            raise ValueError(
                f"Incomplete specification for {self.__class__.__name__} query "
                f"(Mandatory arguments: {', '.join(self._query_args)})"
            )
        self._kwargs = kwargs

    def __init_subclass__(cls, args: List[str], objtype: type):
        cls._query_args = args
        cls.object_type = objtype
        return super().__init_subclass__()

    @abstractmethod
    def __iter__(self):
        """ iterate over queried objects (use yield to implemnet this in derived classes)"""
        pass

    @classmethod
    def get_subclasses(cls):
        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @classmethod
    def get_instances(cls, classname, **kwargs):
        # collect instances of the requested class from all suitable query subclasses.
        result = []
        for querytype in cls.get_subclasses():
            if querytype.object_type.__name__ == classname:
                result.extend(list(querytype(**kwargs)))
        return result


class Registry:
    """
    A registry object provides lookup tables of already instantiated objects
    for different siibra classes. For each class, one such instance table is maintained.

    When a Registry is constructed, it parses a siibra configuration for, which is
    a directory of json specifications for predefined siibra objects.
    The default configuration is pulled from a gitlab repository
    maintained by the siibra development team, but different configurations
    can be passed via the use_configuration() method, and extended via the
    extend_configuration() method.

    Objects and instance tables will only be constructed from the json loaders
    once the instances of a particular class are requested for the first time.
    """

    CONFIG_REPOS = [
        ("https://jugit.fz-juelich.de", 3484),
        ("https://gitlab.ebrains.eu", 93),
    ]

    CONFIGURATIONS = [
        GitlabConnector(server, project, "siibra-{}".format(__version__), skip_branchtest=True)
        for server, project in CONFIG_REPOS
    ]

    # map class name to the relative configuration folders
    # where their preconfiguration specifications can be found.
    CONFIGURATION_FOLDERS = {
        'Atlas': 'atlases',
        'Parcellation': 'parcellations',
        'Space': 'spaces',
        'Map': 'maps',
        'ReceptorDensityFingerprint': "features/fingerprints/receptor",
        'ReceptorDensityProfile': "features/profiles/celldensity",
        'CellDensityFingerprint': "features/fingerprints/celldensity",
        'CellDensityProfile': "features/profiles/celldensity",
    }

    CONFIGURATION_EXTENSIONS = []

    # lists of loaders for json specification files
    # found in the siibra configuration, stored per
    # preconfigured class name. These files can
    # loaded and fed to the Factory.from_json
    # to produce the corresponding object.
    spec_loaders: Dict[str, str] = defaultdict(list)

    # InstanceTable objects with already instantiated
    # objects per name of corresponding class.
    instance_tables: Dict[str, InstanceTable] = {}

    def __init__(self):

        # retrieve json spec loaders from the default configuration
        for connector in self.CONFIGURATIONS:
            try:
                for classname, folder in self.CONFIGURATION_FOLDERS.items():
                    loaders = connector.get_loaders(folder, suffix='json')
                    if len(loaders) > 0:
                        self.spec_loaders[classname] = loaders
                break
            except ConnectionError:
                logger.error(f"Cannot load configuration from {str(connector)}")
                *_, last = self.CONFIGURATIONS
                if connector is last:
                    raise NoSiibraConfigMirrorsAvailableException(
                        "Tried all mirrors, none available."
                    )
        else:
            raise RuntimeError("Cannot pull any default siibra configuration.")

        # add additional spec loaders from extension configurations
        for connector in self.CONFIGURATION_EXTENSIONS:
            try:
                for _class, folder in self.CONFIGURATION_FOLDERS.items():
                    self.spec_loaders[_class].extend(
                        connector.get_loaders(folder, suffix='json')
                    )
                break
            except ConnectionError:
                logger.error(f"Cannot connect to configuration extension {str(connector)}")
                continue

        logger.debug(
            "Preconfigurations: "
            + " | ".join(f"{classname}: {len(L)}" for classname, L in self.spec_loaders.items())
        )

    @property
    def classes(self):
        return list(self.spec_loaders.keys())

    @classmethod
    def use_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("conn needs to be an instance of RepositoryConnector or a valid str")
        logger.info(f"Using custom configuration from {str(conn)}")
        cls.CONFIGURATIONS = [conn]
        REGISTRY.__init__()

    @classmethod
    def extend_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("conn needs to be an instance of RepositoryConnector or a valid str")
        logger.info(f"Extending configuration with {str(conn)}")
        cls.CONFIGURATION_EXTENSIONS.append(conn)
        REGISTRY.__init__()

    def get_instances(self, classname, **kwargs):

        if classname not in self.classes:
            logger.warning(f"Registry does not know how to build {classname} instances")
            return []

        if classname not in self.instance_tables:
            # keep here to avoid circular imports!
            from .factory import Factory
            for i, (fname, loader) in enumerate(self.spec_loaders.get(classname, [])):
                # filename is used by Factory to create an object identifier if none is provided.
                obj = Factory.from_json(dict(loader.data, **{'filename': fname}))
                if classname not in [_.__name__ for _ in [obj.__class__] + list(obj.__class__.__bases__)]:
                    logger.error(
                        f"Specification in {fname} resulted in object type "
                        f"{obj.__class__.__name__}, but {classname} was expected."
                    )
                    continue
                if classname not in self.instance_tables:
                    self.instance_tables[classname] = InstanceTable(matchfunc=obj.__class__.match)
                k = obj.key if hasattr(obj, 'key') else obj.__hash__()
                self.instance_tables[classname].add(k, obj)

        return self.instance_tables[classname]

    def __getitem__(self, cls: Type[T]) -> InstanceTable[T]:
        if isinstance(cls, str):
            logger.debug(f"Accessing __getitem__ with str, fallback to __getattr__")
            return self.__getattr__(cls)
        return self.get_instances(cls.__name__)

    def __getattr__(self, classname: str):
        return self.get_instances(classname)


REGISTRY = Registry()
if SIIBRA_USE_CONFIGURATION:
    logger.warning(f"config.SIIBRA_USE_CONFIGURATION defined, use configuration at {SIIBRA_USE_CONFIGURATION}")
    Registry.use_configuration(SIIBRA_USE_CONFIGURATION)
