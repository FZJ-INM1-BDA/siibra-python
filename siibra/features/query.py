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

from .feature import Feature

from .. import logger
from ..commons import Registry
from ..core import AtlasConcept, Dataset

from abc import ABC
from collections import defaultdict


class FeatureQuery(ABC):
    """
    An abstract class for data features extractors.
    """

    _FEATURETYPE = Feature
    _instances = {}
    REGISTRY = Registry()

    def __init__(self, **kwargs):
        logger.debug(f"Initializing query for {self._FEATURETYPE.__name__} features")
        self.features = []

    def __init_subclass__(cls):
        """
        Registers all subclasses of FeatureQuery.
        """
        logger.debug(
            f"New query {cls.__name__} for {cls._FEATURETYPE.__name__} features"
        )
        cls.REGISTRY.add(cls._FEATURETYPE.modality(), cls)

    @classmethod
    def get_features(cls, concept, modality, group_by=None, **kwargs):
        """
        Retrieve data features linked to an atlas concept, by modality.

        Parameters
        ----------
        concept: AtlasConcept
            An atlas concept, typically a Parcellation or Region object.
        modality: FeatureType, list of FeatureType, or 'all'
            See siibra.features.modalities for available modalities.
            if 'all', all feature modalities will be queried.
        group_by: string, default: None
            If "dataset", group features additionally by dataset id.
            If "concept", group features additionally by the exact linked concept.
            Use the latter for example when you query by a parcellation, and want to get
            features grouped by the particular region of that parcellation which they have been attached to.
        """
        if isinstance(modality, str) and modality == 'all':
            querytypes = [FeatureQuery.REGISTRY[m] for m in FeatureQuery.REGISTRY]
        elif isinstance(modality, (list, tuple)):
            querytypes = [FeatureQuery.REGISTRY[m] for m in modality]
        else:
            if not FeatureQuery.REGISTRY.provides(modality):
                raise RuntimeError(
                    f"Cannot query features - no feature extractor known for feature type {modality}."
                )
            querytypes = [FeatureQuery.REGISTRY[modality]]

        result = {}
        for querytype in querytypes:

            hits = []
            for query in FeatureQuery.queries(querytype.modality(), **kwargs):
                hits.extend(query.execute(concept))
            matches = list(set(hits))

            if group_by is None:
                grouped = matches

            elif group_by == "dataset":
                grouped = defaultdict(list)
                for m in matches:
                    idf = m.id if isinstance(m, Dataset) else None
                    grouped[idf].append(m)

            elif group_by == "concept":
                grouped = defaultdict(list)
                for m in matches:
                    grouped[m._match].append(m)

            else:
                raise ValueError(
                    f"Invalid parameter '{group_by}' for the 'group_by' attribute of get_features. "
                    "Valid entries are: 'dataset', 'concept', or None.")

            result[querytype.modality()] = grouped

        # If only one modality was requested, simplify the dictionary
        if len(result) == 1:
            return next(iter(result.values()))
        else:
            return result

    @classmethod
    def queries(cls, modality: str, **kwargs):
        """
        return global query objects for the given modality, remembering unique instances
        to avoid redundant FeatureQuery instantiations.
        """
        instances = []
        args_hash = hash(tuple(sorted(kwargs.items())))
        Querytype = cls.REGISTRY[modality]
        if (Querytype, args_hash) not in cls._instances:
            logger.debug(f"Building new query {Querytype} with args {kwargs}")
            try:
                cls._instances[Querytype, args_hash] = Querytype(**kwargs)
            except TypeError as e:
                logger.error(f"Cannot initialize {Querytype} query: {str(e)}")
                raise (e)
        instances.append(cls._instances[Querytype, args_hash])
        return instances

    def execute(self, concept: AtlasConcept):
        """
        Executes a query for features associated with an atlas object.
        """
        matches = []
        for feature in self.features:
            if feature.match(concept):
                matches.append(feature)
        return matches

    def __str__(self):
        return "\n".join([str(f) for f in self.features])

    def register(self, feature):
        assert isinstance(feature, self._FEATURETYPE)
        self.features.append(feature)

    @classmethod
    def modality(cls):
        return cls._FEATURETYPE.modality()
