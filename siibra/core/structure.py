
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
""" Abstract base class for any kind of brain structure. """

from . import assignment, region as _region


from abc import ABC, abstractmethod
from typing import Tuple, Dict


class BrainStructure(ABC):
    """ Abstract base class for types who can act as a location filter. """

    # cache assignment results at class level
    _ASSIGNMENT_CACHE: Dict[
        Tuple["BrainStructure", "BrainStructure"],
        assignment.AnatomicalAssignment
    ] = {}

    def intersects(self, other: "BrainStructure") -> bool:
        return self.intersection(other) is not None

    def __contains__(self, other: "BrainStructure") -> bool:
        return self.intersection(other) == other

    @abstractmethod
    def intersection(self, other: "BrainStructure") -> "BrainStructure":
        """
        Return the intersection of two locations,
        ie. the other location filtered by this location.
        """
        pass

    def assign(self, other: "BrainStructure") -> assignment.AnatomicalAssignment:
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
