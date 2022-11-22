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


from .datasets import EbrainsDataset

from ..commons import create_key
from ..registry import REGISTRY

import re


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
        shortname: str = None,
        description: str = None,
        modality: str = "",
        publications: list = [],
        ebrains_ids: dict = {},
    ):
        """
        Construct a new atlas concept base object.

        Parameters
        ----------
            identifier : str
                Unique identifier of the parcellation
            name : str
                Human-readable name of the parcellation
            shortname: str
                Shortform of human-readable name (optional)
            description: str
                Textual description of the parcellation
            modality  :  str or None
                Specification of the modality underlying this concept
            ebrains_ids : dict
                Identifiers of EBRAINS entities corresponding to this Parcellation.
                Key: EBRAINS KG schema, value: EBRAINS KG @id
            publications: list
                List of publications, each a dictionary with "doi" and/or "citation" fields

        """
        self._id = identifier
        self.name = name
        self.shortname = shortname
        self.modality = modality
        self.description = description
        self.publications = publications
        self.datasets = []
        for kg_schema, kg_id in ebrains_ids.items():
            if kg_schema == "minds/core/dataset/v1.0.0":
                if kg_id not in REGISTRY[EbrainsDataset]:
                    REGISTRY[EbrainsDataset].add(
                        kg_id, EbrainsDataset(id=kg_id, name=None)
                    )
                self.datasets.append(REGISTRY[EbrainsDataset][kg_id])
            else:
                raise NotImplementedError(f"No object construction available for EBRAINS schemas {kg_schema}.")

    @property
    def id(self):
        # allows derived classes to assign the id dynamically
        return self._id

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

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}"

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
