# Copyright 2018-2026
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
"""Base interface of the providers that give access to geometry data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np


class GeometryProvider(ABC):
    """
    Access to the data of a geometry in one source format.

    Subclasses declaring a `srctype` are registered, so that configurations
    can refer to them by that name, e.g. ``"providers": {"trx": <url>}``.
    """

    _SUBCLASSES: List[Type[GeometryProvider]] = []

    srctype: str = None

    def __init_subclass__(cls, srctype: str = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if srctype is not None:
            cls.srctype = srctype
            GeometryProvider._SUBCLASSES.append(cls)

    @property
    def fragments(self) -> List[str]:
        """Names of the fragments this provider can deliver separately."""
        return []

    @abstractmethod
    def fetch(self, fragments: Optional[List[str]] = None) -> Any:
        """
        Load the data, in the provider's own representation.

        Parameters
        ----------
        fragments: list of str, default: None
            Exact names of the fragments to load. None loads all data.
        """
        raise NotImplementedError

    @abstractmethod
    def get_coordinates(self, fragments: Optional[List[str]] = None) -> np.ndarray:
        """
        Return the coordinates of all vertices as an (N, 3) array, as stored.

        Parameters
        ----------
        fragments: list of str, default: None
            Exact names of the fragments to include. None includes all data.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _url(self) -> Union[str, Dict[str, str]]:
        """
        The source this provider reads, as configured. Applications such as
        siibra-api use it to access the resource directly.
        """
        raise NotImplementedError
