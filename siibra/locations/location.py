# Copyright 2018-2025
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
"""Concepts that have primarily spatial meaning."""

from __future__ import annotations

from typing import Union, Dict
from abc import abstractmethod

import numpy as np

from ..core.structure import BrainStructure
from ..core import space as _space


class Location(BrainStructure):
    """
    Abstract base class for locations in a given reference space.
    """

    # backend for transforming coordinates between spaces
    # for detail, see https://github.com/HumanBrainProject/hbp-spatial-backend
    SPACEWARP_SERVER = "https://siibra-spatial-backend.apps.ebrains.eu/v1"

    # lookup of space identifiers to be used by SPACEWARP_SERVER
    SPACEWARP_IDS = {
        "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2": "MNI 152 ICBM 2009c Nonlinear Asymmetric",
        "minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992": "MNI Colin 27",
        "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588": "Big Brain (Histology)",
    }

    # The id of BigBrain reference space
    BIGBRAIN_ID = "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588"
    _MASK_MEMO = {}  # cache region masks for Location._assign_region()
    _ASSIGNMENT_CACHE = {}  # caches assignment results, see Region.assign()

    def __init__(self, spacespec: Union[str, Dict[str, str], _space.Space]):
        self._space_spec = spacespec
        self._space_cached = None

    @property
    def space(self) -> _space.Space:
        if self._space_cached is None:
            if self._space_spec is None:
                return None
            elif isinstance(self._space_spec, _space.Space):
                self._space_cached = self._space_spec
            elif isinstance(self._space_spec, dict):
                spec = self._space_spec.get("@id") or self._space_spec.get("name")
                self._space_cached = _space.Space.get_instance(spec)
            elif isinstance(self._space_spec, str):
                self._space_cached = _space.Space.get_instance(self._space_spec)
            else:
                raise ValueError(f"Invalid space spec type: '{type(self._space_spec)}'")
        return self._space_cached

    @abstractmethod
    def warp(self, space: Union[str, "_space.Space"]):
        """Generates a new location by warping the
        current one into another reference space."""
        pass

    @abstractmethod
    def transform(self, affine: np.ndarray, space: Union[str, "_space.Space", None] = None):
        """Returns a new location obtained by transforming the
        reference coordinates of this one with the given affine matrix.

        Parameters
        ----------
        affine : numpy 4x4 ndarray
            affine matrix
        space : reference space (id, name, or Space)
            Target reference space which is reached after
            applying the transform. Note that the consistency
            of this cannot be checked and is up to the user.
        """
        pass

    @property
    def species(self):
        return None if self.space is None else self.space.species

    def __str__(self):
        space_str = "" if self.space is None else f" in {self.space.name}"
        coord_str = "" if len(self) == 0 else f" [{','.join(str(pt) for pt in iter(self))}]"
        return f"{self.__class__.__name__}{space_str}{coord_str}"

    def __repr__(self):
        spacespec = f"'{self.space.id}'" if self.space else None
        return f"<{self.__class__.__name__}({[point.__repr__() for point in self]}), space={spacespec}>"

    def __hash__(self) -> int:
        return hash(self.__repr__())

    @abstractmethod
    def __eq__(self):
        """Required to provide comparison and making the object hashable"""
        raise NotImplementedError

    @staticmethod
    def union(loc0: 'Location', loc1: 'Location') -> 'Location':
        """
        Reassigned at the locations module level for static typing and to avoid
        circular imports. See siibra.locations.__init__.reassign_union()
        """
        raise NotImplementedError(
            "This method is designed to be reassigned at the module level"
        )
