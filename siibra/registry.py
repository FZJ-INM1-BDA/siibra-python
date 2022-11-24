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
from .core.atlas import Atlas
from .core.parcellation import Parcellation, ParcellationVersion
from .core.space import Space
from .core.region import Region
from .core.datasets import EbrainsDataset
from .core.location import Point, PointSet
from .volumes.volume import Volume
from .retrieval.repositories import GitlabConnector, RepositoryConnector

from .retrieval.exceptions import (
    NoSiibraConfigMirrorsAvailableException,
    TagNotFoundException,
)

from typing import Any, Generic, Iterable, Iterator, List, Type, TypeVar, Union, Tuple, Dict, Set, ClassVar
from os import path
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from requests.exceptions import ConnectionError


class Factory:

    @classmethod
    def build_atlas(cls, spec):
        assert spec.get("@type") == "juelich/iav/atlas/v1.0.0"
        atlas = Atlas(
            spec["@id"],
            spec["name"],
            species=spec["species"]
        )
        for space_id in spec["spaces"]:
            atlas._register_space(space_id)
        for parcellation_id in spec["parcellations"]:
            atlas._register_parcellation(parcellation_id)
        return atlas

    @classmethod
    def build_space(cls, spec):
        assert spec.get("@type") == "siibra/space/v0.0.1"
        volumes = list(map(cls.build_volume, spec.get("volumes", [])))
        return Space(
            identifier=spec["@id"],
            name=spec["name"],
            volumes=volumes,
            shortname=spec.get("shortName", ""),
            description=spec.get("description"),
            modality=spec.get("modality"),
            publications=spec.get("publications", []),
            ebrains_ids=spec.get("ebrains", {})
        )

    @classmethod
    def build_region(cls, spec):
        return Region(
            name=spec["name"],
            children=map(cls.build_region, spec.get("children", [])),
            shortname=spec.get("shortname", ""),
            description=spec.get("description", ""),
            publications=spec.get("publications", []),
            ebrains_ids=spec.get("ebrains_ids", {})
        )

    @classmethod
    def build_parcellation(cls, spec):
        assert spec.get("@type", None) == "siibra/parcellation/v0.0.1"
        regions = []
        for regionspec in spec.get("regions", []):
            try:
                regions.append(cls.build_region(regionspec))
            except Exception as e:
                print(regionspec)
                raise e
        parcellation = Parcellation(
            identifier=spec["@id"],
            name=spec["name"],
            regions=regions,
            shortname=spec.get("shortName", ""),
            description=spec.get("description", ""),
            modality=spec.get('modality', ""),
            publications=spec.get("publications", []),
            ebrains_ids=spec.get("ebrains", {}),
        )

        # add version object, if any is specified
        versionspec = spec.get('@version', None)
        if versionspec is not None:
            version = ParcellationVersion(
                name=versionspec.get("name", None),
                parcellation=parcellation,
                collection=versionspec.get("collectionName", None),
                prev_id=versionspec.get("@prev", None),
                next_id=versionspec.get("@next", None),
                deprecated=versionspec.get("deprecated", False)
            )
            parcellation.version = version

        return parcellation

    @classmethod
    def build_volume(cls, spec):
        assert spec.get("@type", None) == "siibra/volume/v0.0.1"
        return Volume(
            name=spec.get("name", ""),
            space_info=spec.get("space", {}),
            urls=spec.get("urls", {})
        )

    @classmethod
    def build_ebrains_dataset(cls, spec):
        assert spec.get("@type", None) == "siibra/snapshots/ebrainsquery/v1"
        return EbrainsDataset(
            id=spec["id"],
            name=spec["name"],
            embargo_status=spec["embargoStatus"],
            cached_data=spec,
        )

    @classmethod
    def build_point(cls, spec):
        assert spec["@type"] == "https://openminds.ebrains.eu/sands/CoordinatePoint"
        # require space spec
        space_id = spec["coordinateSpace"]["@id"]

        # require a 3D point spec for the coordinates
        assert all(c["unit"]["@id"] == "id.link/mm" for c in spec["coordinates"])

        # build the Point
        return Point(
            list(np.float16(c["value"]) for c in spec["coordinates"]),
            space_id=space_id,
        )

    @classmethod
    def build_pointset(cls, spec):
        assert spec["@type"] == "tmp/poly"

        # require space spec
        space_id = spec["coordinateSpace"]["@id"]

        # require mulitple 3D point specs
        coords = []
        for coord in spec["coordinates"]:
            assert all(c["unit"]["@id"] == "id.link/mm" for c in coord)
            coords.append(list(np.float16(c["value"]) for c in coord))

        # build the Point
        return PointSet(coords, space_id=space_id)

    @classmethod
    def from_json(cls, spec: dict):

        if isinstance(spec, str):
            if path.isfile(spec):
                with open(spec, "r") as f:
                    spec = json.load(f)
            else:
                spec = json.loads(spec)

        spectype = spec.get("@type", None)

        if spectype == "juelich/iav/atlas/v1.0.0":
            return cls.build_atlas(spec)
        elif spectype == "siibra/space/v0.0.1":
            return cls.build_space(spec)
        elif spectype == "siibra/parcellation/v0.0.1":
            return cls.build_parcellation(spec)
        elif spectype == "siibra/volume/v0.0.1":
            return cls.build_volume(spec)
        elif spectype == "siibra/space/v0.0.1":
            return cls.build_space(spec)
        elif spectype == "siibra/snapshots/ebrainsquery/v1":
            return cls.build_ebrains_dataset(spec)
        elif spectype == "https://openminds.ebrains.eu/sands/CoordinatePoint":
            return cls.build_point(spec)
        elif spectype == "tmp/poly":
            return cls.build_pointset(spec)
        else:
            raise RuntimeError(f"No factory method for specification type {spectype}.")


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


class Registry:

    CONFIG_REPOS = [
        ("https://jugit.fz-juelich.de", 3484),
        ("https://gitlab.ebrains.eu", 93),
    ]

    CONFIGURATIONS = [
        GitlabConnector(server, project, "siibra-{}".format(__version__), skip_branchtest=True)
        for server, project in CONFIG_REPOS
    ]

    CONFIGURATION_EXTENSIONS = []

    CONFIGURATION_FOLDERS = {
        "atlases": Atlas,
        "parcellations": Parcellation,
        "spaces": Space,
    }

    spec_loaders = defaultdict(list)
    instance_tables = {}

    @classmethod
    def use_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("conn needs to be an instance of RepositoryConnector or a valid str")
        logger.info(f"Using custom configuration from {str(conn)}")
        cls.CONFIGURATIONS = [conn]

    @classmethod
    def extend_configuration(cls, conn: Union[str, RepositoryConnector]):
        if isinstance(conn, str):
            conn = RepositoryConnector._from_url(conn)
        if not isinstance(conn, RepositoryConnector):
            raise RuntimeError("conn needs to be an instance of RepositoryConnector or a valid str")
        logger.info(f"Extending configuration with {str(conn)}")
        cls.CONFIGURATION_EXTENSIONS.append(conn)

    def __init__(self):

        # retrieve json spec loaders main configuration
        for connector in self.CONFIGURATIONS:
            try:
                for folder, _class in self.CONFIGURATION_FOLDERS.items():
                    self.spec_loaders[_class] = connector.get_loaders(folder, suffix='json')
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

        # add spec loaders from extension configurations
        for connector in self.CONFIGURATION_EXTENSIONS:
            try:
                for folder, _class in self.CONFIGURATION_FOLDERS.items():
                    self.spec_loaders[_class].extend(
                        connector.get_loaders(folder, suffix='json')
                    )
                break
            except ConnectionError:
                logger.error(f"Cannot connect to configuration extension {str(connector)}")
                continue

        for _class, loaders in self.spec_loaders.items():
            print(f"{len(loaders):5} specs for {_class.__name__}")

    def get_instances(self, requested_class):
        if requested_class not in self.instance_tables:
            self.instance_tables[requested_class] = InstanceTable(matchfunc=requested_class.match)
            for fname, loader in self.spec_loaders.get(requested_class):
                obj = Factory.from_json(loader.data)
                if not isinstance(obj, requested_class):
                    logger.error(
                        f"Could not instantiate {requested_class} object from {fname}."
                    )
                    continue
                k = obj.key if hasattr(obj, 'key') else obj.__hash__()
                self.instance_tables[requested_class].add(k, obj)
        return self.instance_tables[requested_class]

    def __getattr__(self, classname: str):
        known_classes = {
            _.__name__: _ for _ in 
            set(self.CONFIGURATION_FOLDERS.values()) 
            | set(self.instance_tables.keys())
        }
        return self.get_instances(known_classes.get(classname))

REGISTRY = Registry()