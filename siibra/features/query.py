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
from ..registry import REGISTRY

from abc import ABC
from typing import List


class FeatureQuery(ABC):
    """
    Base class for dynamic data feature queries.
    """

    _FEATURETYPE = Feature
    _parameters = []

    def __init__(self, **kwargs):
        parstr = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        if parstr:
            parstr = "with parameters " + parstr
        logger.info(f"Initializing query for {self._FEATURETYPE.__name__} features {parstr}")
        assert all(p in kwargs for p in self._parameters)
        self.features: List[Feature] = []

    def __init_subclass__(cls, parameters: List[str]):
        """
        Registers new query types in siibra's object registry.
        """
        logger.debug(
            f"New query {cls.__name__} for {cls._FEATURETYPE.__name__} features, parameterized by {parameters}"
        )
        cls._parameters = parameters
        REGISTRY.register_object_query(cls, cls._FEATURETYPE)
        return super().__init_subclass__()

    def __str__(self):
        return "\n".join([str(f) for f in self.features])

    def add_feature(self, feature: Feature):
        assert isinstance(feature, self._FEATURETYPE)
        self.features.append(feature)

    @classmethod
    def modality(cls):
        return cls._FEATURETYPE.modality()
