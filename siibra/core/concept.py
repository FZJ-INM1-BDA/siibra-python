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

from typing import Dict, Union
from pydantic.main import BaseModel
from .datasets import Dataset
import enum

from .. import QUIET, __version__
from ..retrieval import GitlabConnector
from ..commons import TypedRegistry, logger, Registry

import os
import re


# Until openminds is fully supported, we get configurations of siibra concepts from gitlab.
GITLAB_PROJECT_TAG = os.getenv(
    "SIIBRA_CONFIG_GITLAB_PROJECT_TAG", "siibra-{}".format(__version__)
)
USE_DEFAULT_PROJECT_TAG = "SIIBRA_CONFIG_GITLAB_PROJECT_TAG" not in os.environ


_BOOTSTRAP_CONNECTORS = (
    # we use an iterator to only instantiate the one[s] used
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
)


def provide_registry(cls):
    """Used for decorating derived classes - will add a registry of bootstrapped instances then."""

    # find a suitable connector that is reachable
    for connector in _BOOTSTRAP_CONNECTORS:
        try:
            loaders = connector.get_loaders(
                cls._bootstrap_folder,
                ".json",
                progress=f"Bootstrap: {cls.__name__:15.15}",
            )
            break
        except Exception as e:
            print(str(e))
            logger.error(
                f"Cannot connect to configuration server {str(connector)}, trying a different mirror"
            )
            raise (e)
    else:
        # we get here only if the loop is not broken
        raise RuntimeError(
            f"Cannot initialize atlases: No configuration data found for '{GITLAB_PROJECT_TAG}'."
        )

    cls.REGISTRY = Registry(matchfunc=cls.match_spec)
    extensions = []
    with QUIET:
        for fname, loader in loaders:
            logger.info(f"Loading {fname}")
            obj = cls._from_json(loader.data)
            if obj.extends is not None:
                extensions.append(obj)
                continue
            if isinstance(obj, cls):
                cls.REGISTRY.add(obj.key, obj)
            else:
                raise RuntimeError(
                    f"Could not generate object of type {cls} from configuration {fname} - construction provided type {obj.__class__}"
                )

    for e in extensions:
        target = cls.REGISTRY[e.extends]
        target._extend(e)

    return cls

class RegistrySrc(str, enum.Enum):
    GITLAB = 'gitlab'
    EMPTY = 'empty'

def verify_cls(registry_src: RegistrySrc, cls):
    assert issubclass(cls, BaseModel), f'Expecting openminds registry to be subclassing pydantic.BaseModel'
    if registry_src == RegistrySrc.GITLAB:
        assert hasattr(cls, 'parse_legacy') and callable(getattr(cls, 'parse_legacy')), f'For legacy gitlab src, cls must implement static method parse_legacy'
    elif registry_src == RegistrySrc.EMPTY:
        pass
    else:
        raise RuntimeError(f'openminds registry class not yet registered')

main_openminds_registry = Registry()

def provide_openminds_registry(
    bootstrap_folder: str=None,
    registry_src: RegistrySrc=RegistrySrc.GITLAB,
    get_aliases=lambda key, obj: key):

    def provide_rg(cls):
        # sanity check cls construction
        verify_cls(registry_src=registry_src, cls=cls)


        def default_match_spec_fn(obj: cls, spec: Union[str, cls, Dict[str, str]]) -> bool:
            assert isinstance(obj, cls)

            if isinstance(spec, cls):
                return spec.id == obj.id
            if isinstance(spec, dict) and spec.get('@id'):
                return obj.id == spec.get('@id')
            if isinstance(spec, str):
                if hasattr(obj, 'key') and spec == obj.key:
                    return True
                elif spec == obj.id:
                    return True
                else:
                    # match the name
                    words = [w for w in re.split("[ -]", spec)]
                    squeezedname = obj.name.lower().replace(" ", "") if obj.name else ""
                    return any(
                        [
                            all(w.lower() in squeezedname for w in words),
                            spec.replace(" ", "") in squeezedname,
                        ]
                    )
            return False

        registry = TypedRegistry[cls](
            matchfunc=default_match_spec_fn,
            get_aliases=cls.get_aliases if hasattr(cls, 'get_aliases') and callable(cls.get_aliases) else get_aliases
        )
        extensions = []

        def process_instance(instance: cls, fname: str):

            if hasattr(instance, 'extends') and getattr(instance, 'extends'):
                extensions.append(instance)
                return
            if isinstance(instance, cls):
                registry.add(instance.id, instance)

                # also add item to main registry
                if instance.id in main_openminds_registry:
                    logger.warning(f'adding to main registry warning: {instance.id} already exists in main registry. Overwriting...')
                main_openminds_registry.add(instance.id, instance)
            elif isinstance(instance, BaseModel):
                # sometimes, parse_legacy also produces artefacts that does not belong to the class
                # e.g. volume created from space
                main_openminds_registry.add(instance.id, instance)
            else:
                raise RuntimeError(
                    f"Could not generate object of type {cls} from configuration {fname} - construction provided type {instance.__class__}"
                )

        def init_bootstrap():
            # find a suitable connector that is reachable
            if registry_src == RegistrySrc.GITLAB:
                for connector in _BOOTSTRAP_CONNECTORS:
                    try:
                        logger.debug('connecting to {}'.format(bootstrap_folder))
                        loaders = connector.get_loaders(
                            bootstrap_folder,
                            ".json",
                            progress=f"Bootstrap: {cls.__name__:15.15}",
                        )
                        break
                    except Exception as e:
                        print(str(e))
                        logger.error(
                            f"Cannot connect to configuration server {str(connector)}, trying a different mirror"
                        )
                        raise (e)
                else:
                    # we get here only if the loop is not broken
                    raise RuntimeError(
                        f"Cannot initialize atlases: No configuration data found for '{GITLAB_PROJECT_TAG}'."
                    )

                with QUIET:
                    for fname, loader in loaders:
                        logger.info(f"Loading {fname}")
                        obj = cls.parse_legacy(loader.data)
                        if isinstance(obj, list):
                            for inst in obj:
                                process_instance(inst, fname)
                        else:
                            process_instance(obj, fname)

        cls.REGISTRY = registry
        cls.init_bootstrap = init_bootstrap
        return cls
    return provide_rg


class AtlasConcept:
    """
    Parent class encapsulating commonalities of the basic siibra concept like atlas, parcellation, space, region.
    These concepts have an id, name, and key, and they are bootstrapped from metadata stored in an online resources.
    Typically, they are linked with one or more datasets that can be retrieved from the same or another online resource,
    providing data files or additional metadata descriptions on request.
    """

    logger.debug(f"Configuration: {GITLAB_PROJECT_TAG}")
    _bootstrap_folder = None

    @property
    def is_legacy(self):
        return not isinstance(self, BaseModel)

    def __init__(self, identifier, name, dataset_specs=[]):
        if self.is_legacy:
            self.name = name
            self.id = identifier
            self.key = __class__._create_key(name)
            # objects for datasets wil only be generated lazily on request
            self._dataset_specs = dataset_specs
            self._datasets_cached = None
            # this attribute can be used to mark a concept as an extension of another one
            self.extends = None

    def __init_subclass__(cls, type_id=None, bootstrap_folder=None):
        """
        This method is called whenever SiibraConcept gets subclassed
        (see https://docs.python.org/3/reference/datamodel.html)
        """
        logger.debug(
            f"New subclass to {__class__.__name__}: {cls.__name__} (config folder: {bootstrap_folder})"
        )
        cls.type_id = type_id
        if bootstrap_folder is not None:
            cls._bootstrap_folder = bootstrap_folder

    def add_dataset(self, dataset: Dataset):
        """ Explictly add another dataset object to this atlas concept. """
        self._datasets_cached.append(dataset)

    def _populate_datasets(self):
        self._datasets_cached = []
        for spec in self._dataset_specs:
            type_id = Dataset.extract_type_id(spec)
            Specialist = Dataset.REGISTRY.get(type_id, None)
            if Specialist is None:
                logger.warning(f"No class available for building datasets with type {spec.get('@type',None)}. Candidates were {','.join(Dataset.REGISTRY.keys())}. Specification was: {spec}.")
            else:
                obj = Specialist._from_json(spec)
                logger.debug(
                    f"Built {obj.__class__.__name__} object '{obj}' from dataset specification."
                )
                self._datasets_cached.append(obj)

    def _extend(self, other):
        """
        Some concepts allow to be extended by refined concepts.
        The "@extends" attribute in the configuration is used to indicate this.
        Use this method to implement the actual extension operation.
        """
        raise NotImplementedError(
            f"'{self.__class__.__name__}' does not implement an extension "
            f"mechanism with '{other.__class__.__name__}' types.")

    @property
    def datasets(self):
        if self._datasets_cached is None:
            self._populate_datasets()
        return self._datasets_cached

    @staticmethod
    def _create_key(name):
        """
        Creates an uppercase identifier string that includes only alphanumeric
        characters and underscore from a natural language name.
        """
        return re.sub(
            r" +",
            "_",
            "".join([e if e.isalnum() else " " for e in name]).upper().strip(),
        )

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}"

    @property
    def volumes(self):
        """
        The list of available datasets representing image volumes.
        """
        return [d for d in self.datasets if d.is_image_volume]

    @property
    def has_volumes(self):
        """Returns True, if this concept can provide an image volume."""
        return len(self.volumes) > 0

    @property
    def infos(self):
        """
        List of available datasets representing additional information.
        """
        return [d for d in self.datasets if not d.is_image_volume]

    @property
    def publications(self):
        """List of publications found in info datasets."""
        result = []
        for info in self.infos:
            result.extend(info.publications)
        return result

    @property
    def descriptions(self):
        """List of descriptions found in info datasets."""
        result = []
        for info in self.infos:
            result.append(info.description)
        return result

    @property
    def supported_spaces(self):
        """
        The list of spaces for which volumetric datasets are registered.
        """
        return list({v.space for v in self.volumes})

    def get_volumes(self, space):
        """
        Get available volumes sources in the requested template space.

        Parameters
        ----------
        space : Space or str
            template space or template space specification

        Yields
        ------
        A list of volume sources
        """
        # import pdb
        # pdb.set_trace()
        return [v for v in self.volumes if v.space.matches(space)]

    def matches(self, spec):
        """
        Test if the given specification matches the name, key or id of the concept.
        """
        if isinstance(spec, self.__class__) and (spec == self):
            return True
        if isinstance(spec, str):
            if hasattr(self, 'id') and spec == self.id:
                return True

            if hasattr(self, 'get_aliases') and callable(self.get_aliases):
                return any(
                    all(
                        word.lower() in alias.lower()
                        for word in re.split(r"\W+", spec)
                    )
                    for alias in self.get_aliases(self.id, self)
                )
        return False

    @classmethod
    def match_spec(cls, obj, spec):
        assert isinstance(obj, cls)
        return obj.matches(spec)
