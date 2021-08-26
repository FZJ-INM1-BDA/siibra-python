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
from ..core import AtlasConcept

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
    def get_features(cls, concept, modality, group_by_dataset=False, **kwargs):
        """
        Retrieve data features linked to an atlas concept, by modality.

        Parameters
        ----------
        concept: AtlasConcept
            An atlas concept, typically a Parcellation or Region object.
        modality: FeatureType, or 'all'
            See siibra.features.modalities for available modalities.
            if 'all', all feature modalities will be queried.
        group_by_dataset: bool, default: False
            If True, the resulting features are grouped by dataset
        """
        if isinstance(modality, str) and modality == 'all':
            querytypes = [FeatureQuery.REGISTRY[m] for m in FeatureQuery.REGISTRY]
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

            if group_by_dataset:
                grouped = defaultdict(list)
                for match in matches:
                    grouped[match.dataset_id].append(match)
                result[querytype.modality()] = grouped
            else:
                result[querytype.modality()] = matches

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
            if feature.matches(concept):
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
