# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from abc import ABC
from .feature import Feature
from memoization import cached
from .. import logger

class FeatureQuery(ABC):
    """
    An abstract class for data features extractors, implementing a singleton pattern.
    """

    _FEATURETYPE = Feature

    def __init__(self):
        logger.info(f"Initializing query for {self._FEATURETYPE.__name__} features")
        self.features = []

    def execute(self,atlas):
        """
        Executes a query for features associated with atlas object, 
        taking into account its selection of parcellation and region.
        """
        selection = []
        for feature in self.features:
            if feature.matches(atlas):
                selection.append(feature)
        return selection

    def __str__(self):
        return "\n".join([str(f) for f in self.features])

    def register(self,feature):
        assert(isinstance(feature,self._FEATURETYPE))
        self.features.append(feature)

class FeatureQueryRegistry:
    """
    Provides centralized access to objects from all feature query classes.
    """

    def __init__(self):
        self._classes = defaultdict(list)
        self._instances = {}
        self.modalities = {}
        for cls in FeatureQuery.__subclasses__():
            modality = str(cls._FEATURETYPE).split("'")[1].split('.')[-1]
            self._classes[modality].append(cls)
            self.modalities[modality] = cls._FEATURETYPE

    def queries(self,modality,**kwargs):
        """
        return query objects for the given modality
        """
        instances = []
        args_hash = hash(tuple(sorted(kwargs.items())))
        for cls in self._classes[modality]:
            if (cls,args_hash) not in self._instances:
                logger.debug(f"Building new query {cls} with args {kwargs}")
                try:
                    self._instances[cls,args_hash] = cls(**kwargs)
                except TypeError as e:
                    logger.error(f"Cannot initialize {cls._FEATURETYPE.__name__} query: {str(e)}")
                    continue
            instances.append(self._instances[cls,args_hash]) 
        return instances
