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

from abc import ABC


class FeatureQuery(ABC):
    """
    An abstract class for data features extractors.
    """

    _FEATURETYPE = Feature
    _instances = {}
    REGISTRY = Registry()

    def __init__(self):
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

    def execute(self, selection):
        """
        Executes a query for features associated with atlas object,
        taking into account its selection of parcellation and region.
        """
        matches = []
        for feature in self.features:
            if feature.matches(selection):
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
