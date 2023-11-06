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
"""Handles feature queries that rely on live or on-the-fly calculations."""

from ..commons import logger
from ..features.feature import Feature
from ..core.concept import AtlasConcept

from abc import ABC, abstractmethod
from typing import List


class LiveQuery(ABC):

    # set of mandatory query argument names
    _query_args = []

    def __init__(self, **kwargs):
        parstr = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        if parstr:
            parstr = "with parameters " + parstr
        if not all(p in kwargs for p in self._query_args):
            logger.error(
                f"Incomplete specification for {self.__class__.__name__} query "
                f"(Mandatory arguments: {', '.join(self._query_args)})"
            )
        self._kwargs = kwargs

    def __init_subclass__(cls, args: List[str], FeatureType: type):
        cls._query_args = args
        cls.feature_type = FeatureType
        FeatureType._live_queries.append(cls)
        return super().__init_subclass__()

    @abstractmethod
    def query(self, concept: AtlasConcept, **kwargs) -> List[Feature]:
        raise NotImplementedError(f"Dervied class {self.__class__} needs to implement query()")
        pass
