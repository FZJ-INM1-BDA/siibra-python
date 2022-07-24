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

from .datasets import Dataset
from ..commons import logger, create_key

import re


class AtlasConcept:
    """
    Parent class encapsulating commonalities of the basic siibra concept like atlas, parcellation, space, region.
    These concepts have an id, name, and key, and they are bootstrapped from metadata stored in an online resources.
    Typically, they are linked with one or more datasets that can be retrieved from the same or another online resource,
    providing data files or additional metadata descriptions on request.
    """
    def __init__(self, identifier, name, dataset_specs):
        self.id = identifier
        self.name = name
        # objects for datasets wil only be generated lazily on request
        self._dataset_specs = dataset_specs
        self._datasets_cached = None
        # this attribute can be used to mark a concept as an extension of another one
        self.extends = None

    @property
    def key(self):
        return create_key(self.name)

    def __init_subclass__(cls, type_id=None):
        """
        This method is called whenever AtlasConcept gets subclassed
        (see https://docs.python.org/3/reference/datamodel.html)
        """
        cls.type_id = type_id
        return super().__init_subclass__()

    def add_dataset(self, dataset: Dataset):
        """ Explictly add another dataset object to this atlas concept. """
        self._datasets_cached.append(dataset)

    def _populate_datasets(self):
        self._datasets_cached = []
        for spec in self._dataset_specs:
            type_id = Dataset.extract_type_id(spec)
            Specialist = Dataset.REGISTRY.get(type_id, None)
            if Specialist is None:
                logger.warn(f"No class available for building datasets with type {spec.get('@type',None)}. Candidates were {','.join(Dataset.REGISTRY.keys())}. Specification was: {spec}.")
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

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}"

    @property
    def volumes(self):
        """
        The list of available datasets representing image volumes.
        """
        return [d for d in self.datasets if d.is_volume]

    @property
    def surfaces(self):
        """
        The list of available datasets representing surface volumes.
        """
        return [d for d in self.datasets if d.is_surface]

    @property
    def has_volumes(self):
        """Returns True, if this concept can provide an image volume."""
        return len(self.volumes) > 0

    @property
    def has_surfaces(self):
        """Returns True, if this concept can provide a surface volume."""
        return len(self.surfaces) > 0

    @property
    def infos(self):
        """
        List of available datasets representing additional information.
        """
        return [d for d in self.datasets if not d.is_volume]

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
        return [v for v in self.volumes if v.space.matches(space)]

    def matches(self, spec):
        """
        Test if the given specification matches the name, key or id of the concept.
        """
        if isinstance(spec, self.__class__) and (spec == self):
            return True
        elif isinstance(spec, str):
            if spec == self.key:
                return True
            elif spec == self.id:
                return True
            else:
                # match the name
                words = [w for w in re.split("[ -]", spec)]
                squeezedname = self.name.lower().replace(" ", "")
                return any(
                    [
                        all(w.lower() in squeezedname for w in words),
                        spec.replace(" ", "") in squeezedname,
                    ]
                )
        return False

    @classmethod
    def match(cls, obj, spec):
        """Match a given object specification. """
        assert isinstance(obj, cls)
        return obj.matches(spec)
