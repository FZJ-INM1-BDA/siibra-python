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
from __future__ import annotations

from . import assignment

from ..core import region as _region

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union


class Location(ABC):
    """
    Abstract base class for locations in a given reference space.
    """

    # backend for transforming coordinates between spaces
    SPACEWARP_SERVER = "https://hbp-spatial-backend.apps.hbp.eu/v1"

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

    def __init__(self, space):
        self._space_spec = space
        self._space_cached = None

    @property
    def space(self):
        if self._space_cached is None:
            from ..core.space import Space
            self._space_cached = Space.get_instance(self._space_spec)
        return self._space_cached

    @abstractmethod
    def warp(self, space):
        """Generates a new location by warping the
        current one into another reference space."""
        pass

    @abstractmethod
    def transform(self, affine: np.ndarray, space=None):
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

    def __len__(self):
        """The number of coordinates or sublocations in a location."""
        return 0

    @property
    def species(self):
        return None if self.space is None else self.space.species

    def __str__(self):
        space_str = "" if self.space is None else f" in {self.space.name}"
        coord_str = "" if len(self) == 0 else f" [{','.join(str(l) for l in iter(self))}]"
        return f"{self.__class__.__name__}{space_str}{coord_str}"


class LocationFilter(ABC):
    """ Abstract base class for types who can act as a location filter. """

    # cache assignment results at class level
    _ASSIGNMENT_CACHE: Dict[
        Tuple[Union[Location, "_region.Region"], Union[Location, "_region.Region"]],
        assignment.AnatomicalAssignment
    ] = {}

    def intersects(self, loc: Location) -> bool:
        return self.intersection(loc) is not None

    def __contains__(self, loc: Location) -> bool:
        return self.intersection(loc) == loc

    @abstractmethod
    def intersection(self, other: Location) -> Location:
        """
        Return the intersection of two locations,
        ie. the other location filtered by this location.
        """
        pass

    def assign(self, other: Union[Location, _region.Region]) -> assignment.AnatomicalAssignment:
        """
        Compute assignment of a location to this filter.

        Two cases:
        1) self is location, other is location -> look at spatial intersection/relationship, do it here
        2) self is location, other is region -> get region map, then call again. do it here
        If self is region -> Region overwrite this method, adressed there

        Parameters
        ----------
        other : Location or Region

        Returns
        -------
        assignment.AnatomicalAssignment or None
            None if there is no AssignmentQualification found.
        """
        assert not isinstance(self, _region.Region)  # method is overwritten by Region!
        if (self, other) in self._ASSIGNMENT_CACHE:
            return self._ASSIGNMENT_CACHE[self, other]
        if (other, self) in self._ASSIGNMENT_CACHE:
            return self._ASSIGNMENT_CACHE[other, self].invert()

        if isinstance(other, _region.Region):
            self._ASSIGNMENT_CACHE[self, other] = other.assign(self).invert()
            return self._ASSIGNMENT_CACHE[self, other]
        else:  # other is a location object, just check spatial relationships
            if self == other:
                qualification = assignment.AssignmentQualification.EXACT
            elif self in other:
                qualification = assignment.AssignmentQualification.CONTAINS
            elif other in self:
                qualification = assignment.AssignmentQualification.CONTAINED
            elif self.intersects(other):
                qualification = assignment.AssignmentQualification.OVERLAPS
            else:
                qualification = None
            if qualification is None:
                self._ASSIGNMENT_CACHE[self, other] = None
            else:
                self._ASSIGNMENT_CACHE[self, other] = assignment.AnatomicalAssignment(self, other, qualification)
        return self._ASSIGNMENT_CACHE[self, other]
