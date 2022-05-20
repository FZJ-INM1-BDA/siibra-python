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

from siibra.core.serializable_concept import JSONSerializable
from .feature import Feature

from .. import logger
from ..commons import TypedRegistry
from ..core import AtlasConcept, Dataset

from abc import ABC
from collections import defaultdict
from typing import Dict, List


class FeatureQuery(ABC):
    """
    An abstract class for data features extractors.
    """

    _FEATURETYPE = Feature
    _instances = {}
    _implementations: Dict[str, List['FeatureQuery']] = defaultdict(list)

    def __init__(self, **kwargs):
        logger.debug(f"Initializing query for {self._FEATURETYPE.__name__} features")
        self.features: List[Feature] = []

    def __init_subclass__(cls):
        """
        Registers all subclasses of FeatureQuery.
        """
        logger.debug(
            f"New query {cls.__name__} for {cls._FEATURETYPE.__name__} features"
        )
        cls._implementations[cls._FEATURETYPE.modality()].append(cls)
        return super().__init_subclass__()

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
        # helper func: convert modality spec into unique modality 
        mod = lambda modality: cls.get_modalities()[modality]

        if isinstance(modality, str) and modality == 'all':
            # use union of all featurequery implementations
            querytypes = sum(FeatureQuery._implementations.values(), [])
        elif isinstance(modality, (list, tuple)):
            # use union of featurequery implementations from multiple modalities
            querytypes = sum(
                [FeatureQuery._implementations[mod(m)] for m in modality], []
            )
        else:
            if mod(modality) not in cls._implementations:
                raise RuntimeError(
                    f"No feature query known for feature type {modality}."
                )
            querytypes = cls._implementations[mod(modality)]

        result = {}
        args_hash = hash(tuple(sorted(kwargs.items())))
        for querytype in querytypes:

            if (querytype, args_hash) not in cls._instances:
                logger.debug(f"Building new query {querytype} with args {kwargs}")
                try:
                    cls._instances[querytype, args_hash] = querytype(**kwargs)
                except TypeError as e:
                    logger.error(f"Cannot initialize {querytype} query: {str(e)}")
                    raise (e)

            query = cls._instances[querytype, args_hash]
            matches = query.execute(concept, **kwargs)
            logger.debug(f"{len(matches)} matches from query {query.__class__.__name__}")

            if group_by is None:
                if querytype.modality() not in result:
                    result[querytype.modality()] = []
                result[querytype.modality()].extend(matches)

            elif group_by == "dataset":
                if querytype.modality() not in result:
                    result[querytype.modality()] = defaultdict(list)
                for m in matches:
                    idf = m.id if isinstance(m, Dataset) else None
                    result[querytype.modality()][idf].append(m)

            elif group_by == "concept":
                if querytype.modality() not in result:
                    result[querytype.modality()] = defaultdict(list)
                for m in matches:
                    result[querytype.modality()][m._match].append(m)

            else:
                raise ValueError(
                    f"Invalid parameter '{group_by}' for the 'group_by' attribute of get_features. "
                    "Valid entries are: 'dataset', 'concept', or None.")

        # If only one modality was requested, simplify the dictionary
        if len(result) == 1:
            return next(iter(result.values()))
        else:
            return result

    @classmethod
    def get_feature_by_id(cls, feature_id: str) -> Feature:
        applicable_queries = [query
            for queries in FeatureQuery._implementations.values()
            for query in queries
            if issubclass(query._FEATURETYPE, JSONSerializable)]

        queries_matched = [
            q for q in applicable_queries if feature_id.startswith(q._FEATURETYPE.get_model_type())
        ]

        if len(queries_matched) == 0:
            logger.warn(f"feature_id {feature_id} cannot be properly matched, bruteforce get feature")

        use_queries = queries_matched if len(queries_matched) > 0 else applicable_queries
        for querytype in use_queries:
            for feature in querytype().features:
                assert isinstance(feature, JSONSerializable), f"feature should be an instance of JSONSerializable, but is not"
                if feature.model_id == feature_id:
                    return feature
        return None

    @classmethod
    def get_modalities(cls):
        return TypedRegistry[str](
            elements={
                c:c for c in cls._implementations.keys()
            }
        )

    @classmethod
    def queries(cls, modality: str, **kwargs):
        """
        return global query objects for the given modality, remembering unique instances
        to avoid redundant FeatureQuery instantiations.
        """
        instances = []
        args_hash = hash(tuple(sorted(kwargs.items())))
        matched_modality = cls.get_modalities()[modality]
        for Querytype in cls._implementations[matched_modality]:
            if (Querytype, args_hash) not in cls._instances:
                logger.debug(f"Building new query {Querytype} with args {kwargs}")
                try:
                    cls._instances[Querytype, args_hash] = Querytype(**kwargs)
                except TypeError as e:
                    logger.error(f"Cannot initialize {Querytype} query: {str(e)}")
                    raise (e)
            instances.append(cls._instances[Querytype, args_hash])
        return instances

    def execute(self, concept: AtlasConcept, **kwargs):
        """
        Executes a query for features associated with an atlas object.
        """
        matches = []
        for feature in self.features:
            if feature.match(concept, **kwargs):
                matches.append(feature)        
        return matches

    def __str__(self):
        return "\n".join([str(f) for f in self.features])

    def register(self, feature: Feature):
        assert isinstance(feature, self._FEATURETYPE)
        self.features.append(feature)

    @classmethod
    def modality(cls):
        return cls._FEATURETYPE.modality()
