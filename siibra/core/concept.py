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
"""Parent class to siibra main concepts."""
from ..commons import (
    create_key,
    clear_name,
    logger,
    InstanceTable,
    Species,
    TypePublication
)
from ..retrieval import cache

import re
from typing import TypeVar, Type, Union, List, TYPE_CHECKING, Dict

T = TypeVar("T", bound="AtlasConcept")
_REGISTRIES: Dict[Type[T], InstanceTable[T]] = {}


@cache.Warmup.register_warmup_fn(is_factory=True)
def _atlas_concept_warmup():
    return [cls.registry for cls in _REGISTRIES]


if TYPE_CHECKING:
    from ..retrieval.datasets import EbrainsDataset
    TypeDataset = EbrainsDataset


def get_registry(subclass_name: str):
    subclasses = {c.__name__: c for c in _REGISTRIES}
    if subclass_name in subclasses:
        return subclasses[subclass_name].registry()
    else:
        logger.warn(f"No registry for atlas concepts named {subclass_name}")
        return None


class AtlasConcept:
    """
    Parent class encapsulating commonalities of the basic siibra concept like atlas, parcellation, space, region.
    These concepts have an id, name, and key, and they are bootstrapped from metadata stored in an online resources.
    Typically, they are linked with one or more datasets that can be retrieved from the same or another online resource,
    providing data files or additional metadata descriptions on request.
    """

    def __init__(
        self,
        identifier: str,
        name: str,
        species: Union[str, Species],
        shortname: str = None,
        description: str = None,
        modality: str = "",
        publications: List[TypePublication] = [],
        datasets: List['TypeDataset'] = [],
        spec=None
    ):
        """
        Construct a new atlas concept base object.

        Parameters
        ----------
        identifier : str
            Unique identifier of the parcellation
        name : str
            Human-readable name of the parcellation
        species: Species or string
            Specification of the species
        shortname: str
            Shortform of human-readable name (optional)
        description: str
            Textual description of the parcellation
        modality  :  str or None
            Specification of the modality underlying this concept
        datasets : list
            list of datasets corresponding to this concept
        publications: list
            List of publications, each a dictionary with "doi" and/or "citation" fields
        spec: dict, default: None
            The preconfigured specification.
        """
        self._id = identifier
        self.name = name
        self._species_cached = None if species is None \
            else Species.decode(species)  # overwritable property implementation below
        self.shortname = shortname
        self.modality = modality
        self._description = description
        self._publications = publications
        self.datasets = datasets
        self._spec = spec
        self._CACHED_MATCHES = {}  # we cache match() function results

    @property
    def description(self):
        if self._description:
            return self._description
        for ds in self.datasets:
            if ds.description:
                return ds.description
        return ''

    @property
    def LICENSE(self) -> str:
        licenses = {ds.LICENSE for ds in self.datasets if ds.LICENSE}
        if not licenses:
            return "No license information is found."
        if len(licenses) == 1:
            return next(iter(licenses))
        logger.info("Found multiple licenses corresponding to datasets.")
        return '\n'.join(licenses)

    @property
    def urls(self) -> List[str]:
        """The list of URLs (including DOIs) associated with this atlas concept."""
        return [
            url.get("url")
            for ds in self.datasets
            for url in ds.urls
        ]

    @property
    def authors(self):
        return [
            contributer['name']
            for ds in self.datasets
            for contributer in ds.contributors
        ]

    @property
    def publications(self) -> List[TypePublication]:
        return [
            *self._publications,
            *[
                {'citation': f"Dataset name: {ds.name}", 'url': url.get("url")}
                for ds in self.datasets
                for url in ds.urls
            ]
        ]

    @property
    def species(self) -> Species:
        # Allow derived classes to implement a lazy loader (e.g. in Map)
        if self._species_cached is None:
            raise RuntimeError(f"No species defined for {self}.")
        return self._species_cached

    @classmethod
    def registry(cls: Type[T]) -> InstanceTable[T]:
        if cls._configuration_folder is None:
            return None
        if _REGISTRIES[cls] is None:
            from ..configuration import Configuration
            conf = Configuration()
            # visit the configuration to provide a cleanup function
            # in case the user changes the configuration during runtime.
            Configuration.register_cleanup(cls.clear_registry)
            assert cls._configuration_folder in conf.folders
            objects = conf.build_objects(cls._configuration_folder)
            logger.debug(f"Built {len(objects)} preconfigured {cls.__name__} objects.")
            assert len(objects) > 0
            assert all([hasattr(o, 'key') for o in objects])

            # TODO Map.registry() returns InstanceTable that contains two different types, SparseMap and Map
            # Since we take the objects[0].__class__.match, if the first element happen to be SparseMap, this could result.
            # Code to reproduce:
            """
            import siibra
            r = siibra.volumes.Map.registry()
            """
            if len({o.__class__ for o in objects}) > 1:
                logger.warning(
                    f"{cls.__name__} registry contains multiple classes: "
                    f"{', '.join(list({o.__class__.__name__ for o in objects}))}"
                )
            assert hasattr(objects[0].__class__, "match") and callable(objects[0].__class__.match)
            _REGISTRIES[cls] = InstanceTable(
                elements={o.key: o for o in objects},
                matchfunc=objects[0].__class__.match
            )
        return _REGISTRIES[cls]

    @classmethod
    def clear_registry(cls):
        _REGISTRIES[cls] = None

    @classmethod
    def get_instance(cls, spec: str):
        """
        Parameters
        ----------
            spec: str
                Specification of the class the instance is requested.
        Returns
        -------
            an instance of this class matching the given specification from its
            registry if possible, otherwise None.
        Raises
        ------
            IndexError
                If spec cannot match any instance
        """
        if cls.registry() is not None:
            return cls.registry().get(spec)

    @property
    def id(self):
        # allows derived classes to assign the id dynamically
        return self._id

    @property
    def key(self):
        return create_key(self.name)

    def __init_subclass__(cls, configuration_folder: str = None):
        """
        This method is called whenever AtlasConcept gets subclassed
        (see https://docs.python.org/3/reference/datamodel.html)
        """
        cls._configuration_folder = configuration_folder
        _REGISTRIES[cls] = None
        return super().__init_subclass__()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<{self.__class__.__name__}(identifier='{self.id}', name='{self.name}', species='{self.species}')>"

    def matches(self, spec) -> bool:
        """
        Parameters
        ----------
        spec: str
            Specification checked within the concept name, key or id

        Returns
        -------
        bool
            Whether the given specification matches the name, key or id of the concept.
        """
        if spec not in self._CACHED_MATCHES:
            self._CACHED_MATCHES[spec] = False
            if isinstance(spec, self.__class__) and (spec == self):
                self._CACHED_MATCHES[spec] = True
            elif isinstance(spec, str):
                if spec == self.key:
                    self._CACHED_MATCHES[spec] = True
                elif spec == self.id:
                    self._CACHED_MATCHES[spec] = True
                else:
                    # match the name
                    words = [w for w in re.split("[ -]", spec)]
                    squeezedname = clear_name(self.name.lower()).replace(" ", "")
                    self._CACHED_MATCHES[spec] = any(
                        [
                            all(w.lower() in squeezedname for w in words),
                            spec.replace(" ", "") in squeezedname,
                        ]
                    )
        return self._CACHED_MATCHES[spec]

    @classmethod
    def match(cls, obj, spec) -> bool:
        """Match a given object specification. """
        assert isinstance(obj, cls)
        return obj.matches(spec)

    def __gt__(self, other: 'AtlasConcept'):
        """
        Compare this atlas concept with other atlas concepts of the same kind
        with it's name.
        """
        if self.__class__ is not other.__class__:
            raise ValueError("Cannot compare different atlas concept types.")
        return self.name > other.name

    def __lt__(self, other: 'AtlasConcept'):
        """
        Compare this atlas concept with other atlas concepts of the same kind
        with it's name.
        """
        if self.__class__ is not other.__class__:
            raise ValueError("Cannot compare different atlas concept types.")
        return self.name < other.name
