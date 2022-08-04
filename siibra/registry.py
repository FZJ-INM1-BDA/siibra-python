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
from .retrieval.repositories import (
    GitlabConnector,
    RepositoryConnector,
)
from .retrieval.exceptions import (
    NoSiibraConfigMirrorsAvailableException,
    TagNotFoundException,
)
from .config import (
    USE_DEFAULT_PROJECT_TAG,
    GITLAB_PROJECT_TAG,
    SIIBRA_USE_LOCAL_REPOSITORY,
)

import os
from typing import Any, Generic, Iterable, Iterator, List, Type, TypeVar, Union, Tuple
from collections import defaultdict


T = TypeVar("T")


class TypedObjectLUT(Generic[T], Iterable):
    """
    Lookup table for objects by names. Provide attribute-access and iteration to a set of named elements,
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
            self._elements = {}
        else:
            assert isinstance(elements, dict)
            assert all(isinstance(k, str) for k in elements.keys())
            self._elements = elements
        self._matchfunc = matchfunc

    def add(self, key: Union[str, int], value: T) -> None:
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

    def __sub__(self, obj) -> "TypedObjectLUT[T]":
        """
        remove an object from the registry
        """
        if obj in self._elements.values():
            return TypedObjectLUT[T](
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


class ObjectLUT(TypedObjectLUT[Any]):
    pass


class ObjectRegistry:
    """
    Registry of preconfigured and dynamically retrievable objects
     of differenct siibra classes.
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

    _preconfiguration_folders = {}
    _dynamic_query_types = defaultdict(set)
    _dynamic_queries = defaultdict(set)

    # TODO memory management concern, esp in siibra-api
    _objects = {}

    _CONFIGURATIONS = [
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
    ] if SIIBRA_USE_LOCAL_REPOSITORY is None else [ RepositoryConnector._from_url(SIIBRA_USE_LOCAL_REPOSITORY) ]

    if SIIBRA_USE_LOCAL_REPOSITORY is not None:
        logger.warn(f"SIIBRA_USE_LOCAL_REPOSITORY is set, use {SIIBRA_USE_LOCAL_REPOSITORY} as default configurations")
        
    _CONFIGURATION_EXTENSIONS = []

    @classmethod
    def use_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        logger.info(f"Adding configuration {str(conn)}")
        cls._CONFIGURATIONS.insert(0, conn)

    @classmethod
    def extend_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        logger.info(f"Extending configuration with {str(conn)}")
        cls._CONFIGURATION_EXTENSIONS.append(conn)

    @classmethod
    def register_preconfiguration(cls, folder: str, configured_class: type):
        """
        Adds a configuration folder with specifications for bootstrapping
        preconfigured objects of the given class.
        """
        cls._preconfiguration_folders[configured_class] = folder

    @classmethod
    def register_object_query(cls, query: type, queried_class: type):
        """
        Adds a dynamic query type which produces
        objects of the given class when called.
        """
        cls._dynamic_query_types[queried_class].add(query)

    @classmethod
    def preconfigure_instances(cls, registered_cls):

        key = (registered_cls, ())

        # at this point we should not have a registry for this class yet, and will create it.
        assert key not in cls._objects
        cls._objects[key] = TypedObjectLUT[registered_cls](
            matchfunc=registered_cls.match
        )

        # bootstrap preconfigured objects
        if registered_cls in cls._preconfiguration_folders:

            # get object loaders from siibra configuration
            for connector in cls._CONFIGURATIONS:
                try:
                    loaders = connector.get_loaders(
                        cls._preconfiguration_folders[registered_cls],
                        ".json",
                        progress=f"Bootstrap: {registered_cls.__name__:15.15}",
                    )
                    break
                except Exception as e:
                    print(str(e))
                    logger.error(
                        f"Cannot connect to configuration server {str(connector)}"
                    )
                    *_, last = cls._CONFIGURATIONS
                    if connector is last:
                        raise NoSiibraConfigMirrorsAvailableException(
                            "Tried all mirrors, none available."
                        )
            else:
                # we get here only if the loop is not broken
                raise TagNotFoundException(
                    f"Cannot bootstrap '{registered_cls.__name__}' objects: No configuration data found for '{GITLAB_PROJECT_TAG}'."
                )

            num_default_loaders = len(loaders)

            # add configuration extensions
            for connector in cls._CONFIGURATION_EXTENSIONS:
                try:
                    extloaders = connector.get_loaders(
                        cls._preconfiguration_folders[registered_cls],
                        ".json",
                        progress=f"Bootstrap: {registered_cls.__name__:15.15}",
                    )
                except Exception as e:
                    print(str(e))
                    logger.error(f"Cannot connect to configuration extension {str(connector)}")
                    continue
                loaders.extend(extloaders)

            # boostrap the preconfigured objects
            for i, (fname, loader) in enumerate(loaders):
                obj = registered_cls._from_json(loader.data)
                if not isinstance(obj, registered_cls):
                    raise RuntimeError(
                        f"Could not instantiate {registered_cls} object from '{fname}'"
                    )
                objkey = obj.key if hasattr(obj, 'key') else obj.__hash__()
                if i >= num_default_loaders:
                    if objkey in cls._objects[key]:
                        logger.info(
                            f"Extension updates existing {registered_cls.__name__} "
                            f"({os.path.basename(fname)})"
                        )
                    else:
                        logger.info(
                            f"Extension specifies new {registered_cls.__name__} "
                            f"({os.path.basename(fname)})"
                        )
                cls._objects[key].add(objkey, obj)

    @classmethod
    def initialize_queries(cls, registered_cls: type, params: List[Tuple]):

        key = (registered_cls, params)
        if key in cls._dynamic_queries:
            # this cls/params pair has already been initialized
            return

        for Querytype in cls._dynamic_query_types[registered_cls]:
            if not all(p in dict(params) for p in Querytype._parameters):
                raise AttributeError(
                    f"Parameter specification missing for querying '{registered_cls.__name__}' "
                    f"objects (required: {', '.join(Querytype._parameters)})"
                )
            logger.info(f"Building new query {Querytype} with parameters {params}")
            try:
                cls._dynamic_queries[key].add(Querytype(**dict(params)))
            except TypeError as e:
                logger.error(f"Cannot initialize {Querytype} query: {str(e)}")
                raise (e)

    @property
    def known_types(self):
        return set(self._preconfiguration_folders) | set(self._dynamic_query_types)

    def get_objects(self, object_type, **kwargs):
        """
        Retrieve objects by type
        """

        if object_type not in self.known_types:
            logger.warn(f"No objects of type '{object_type.__name__}' known.")
            return None

        params = tuple(sorted(kwargs.items()))
        key = (object_type, params)

        # start with preconfigured objects
        if object_type in self._preconfiguration_folders and key not in self._objects:
            self.preconfigure_instances(object_type)

        if key not in self._objects:
            self._objects[key] = TypedObjectLUT[object_type](
                matchfunc=object_type.match
            )

        # extend by objects from dynamic queries
        self.initialize_queries(object_type, params)
        for Querytype in self._dynamic_query_types[object_type]:
            assert hasattr(Querytype, "_parameters")
            for query in self._dynamic_queries[key]:
                for obj in query.features:
                    objkey = obj.key if hasattr(obj, "key") else obj.__hash__()
                    self._objects[key].add(objkey, obj)

        return self._objects[key]

    def __getattr__(self, classname: str):
        """
        Access predefined object registries by class name, e.g.
        REGISTRY.Atlas.
        For objects which are dynamically produced by parameterized queries,
        a function of the parameters is returned instead of a registry.
        You might want to use get_objects(type, parameters) for those for better readability.
        """
        classnames = {c.__name__: c for c in self.known_types}
        if classname not in classnames:
            raise AttributeError(
                f"No objects of type '{classname}' found. Use one of "
                f"{', '.join(classnames)}."
            )
        return self[classnames[classname]]

    def __getitem__(self, object_type: Type[T]) -> TypedObjectLUT[T]:
        """
        Access predefined object registries by class, e.g.
        REGISTRY[Atlas].
        For objects which are dynamically produced by parameterized queries,
        a function of the parameters is returned instead of a registry.
        You might want to use get_objects(type, parameters) for those for better readability.
        """
        if object_type in self._preconfiguration_folders:
            return self.get_objects(object_type)

        if object_type in self._dynamic_query_types:
            querytypes = self._dynamic_query_types[object_type]
            assert all(hasattr(_, "_parameters") for _ in querytypes)
            params = [p for _ in querytypes for p in _._parameters]
            if len(params) > 0:
                logger.warn(
                    f"Retrieval of '{object_type.__name__}' objects requires parameters "
                    f"({', '.join(params)}). A function of these parameters is returned "
                    "instead of an object lookup table."
                )
                return lambda **kwargs: self.get_objects(object_type, **kwargs)
            else:
                return self.get_objects(object_type)

        raise AttributeError(f"No objects of type '{object_type.__name__}' known.")


class Preconfigure:
    """
    Decorator for preconfiguring instances of siibra classes from siibra configuration files.

    Requires to provide the configuration subfolder which contains json files for bootstrapping
    objects of that class.

    For example,
        @Preconfigure("atlases")
        class Atlas...

    will make inform the PreconfiguredObjects class to provide a registry of predefined "Atlas" objects,
    and (when first requested) bootstrap objects from the "atlases" subfolder of the siibra configuration.
    """

    def match(self, specification):
        """Match a given specification. Defaults to == operator,
        but may be overriden by decorated classes."""
        return self == specification

    FUNCS_REQUIRED = {"_from_json": None, "match": match}

    def __init__(self, folder):
        self.folder = folder

    def __call__(self, cls):

        for fncname, defaultfnc in self.FUNCS_REQUIRED.items():
            if not (hasattr(cls, fncname)):
                if defaultfnc is None:
                    raise TypeError(
                        f"Class '{cls.__name__}' needs to implement '{fncname}' "
                        "in order to use the @preconfigure decorator."
                    )
                else:
                    setattr(cls, fncname, defaultfnc)

        ObjectRegistry.register_preconfiguration(self.folder, cls)

        return cls


REGISTRY = ObjectRegistry()
