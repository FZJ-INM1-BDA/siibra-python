
# Copyright 2018-2023
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
"""
Abstract base class for any kind of brain structure.
A brain structure is more general than a brain region. It refers to any object
defining a spatial extent in one or more reference spaces, and can thus be
used to compute intersections with other structures in space. For example,
a brain region is a structure which is at the same time an AtlasConcept. A
bounding box in MNI space is a structure, but not an AtlasConcept.
"""

from . import assignment, region as _region

from abc import ABC, abstractmethod
from typing import Tuple, Dict


class BrainStructure(ABC):
    """Abstract base class for types who can act as a location filter."""

    # cache assignment results at class level
    _ASSIGNMENT_CACHE: Dict[
        Tuple["BrainStructure", "BrainStructure"],
        "assignment.AnatomicalAssignment"
    ] = {}

    def intersects(self, other: "BrainStructure") -> bool:
        """
        Whether or not two BrainStructures have any intersection.
        """
        return self.intersection(other) is not None

    def __contains__(self, other: "BrainStructure") -> bool:
        return self.intersection(other) == other

    def __hash__(self) -> int:
        return hash(self.__repr__())

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def intersection(self, other: "BrainStructure") -> "BrainStructure":
        """
        Return the intersection of two BrainStructures,
        ie. the other BrainStructure filtered by this BrainStructure.
        """
        pass

    def assign(self, other: "BrainStructure") -> assignment.AnatomicalAssignment:
        """
        Compute assignment of a BrainStructure to this filter.

        Parameters
        ----------
        other : Location or Region

        Returns
        -------
        assignment.AnatomicalAssignment or None
            None if there is no AssignmentQualification found.
        """
        # Two cases:
        # 1) self is location, other is location -> look at spatial intersection/relationship, do it here
        # 2) self is location, other is region -> get region map, then call again. do it here
        # If self is region -> Region overwrite this method, adressed there

        assert not isinstance(self, _region.Region)  # method is overwritten by Region!
        if (self, other) in self._ASSIGNMENT_CACHE:
            return self._ASSIGNMENT_CACHE[self, other]
        if (other, self) in self._ASSIGNMENT_CACHE:
            return self._ASSIGNMENT_CACHE[other, self].invert()

        if isinstance(other, _region.Region):
            inverse_assignment = other.assign(self)
            if inverse_assignment is None:
                self._ASSIGNMENT_CACHE[self, other] = None
            else:
                self._ASSIGNMENT_CACHE[self, other] = inverse_assignment.invert()
            return self._ASSIGNMENT_CACHE[self, other]
        else:  # other is a location object, just check spatial relationships
            qualification = None
            if self == other:
                qualification = assignment.Qualification.EXACT
            else:
                intersection = self.intersection(other)
                if intersection is not None:
                    if intersection == other:
                        qualification = assignment.Qualification.CONTAINS
                    elif intersection == self:
                        qualification = assignment.Qualification.CONTAINED
                    else:
                        qualification = assignment.Qualification.OVERLAPS
            if qualification is None:
                self._ASSIGNMENT_CACHE[self, other] = None
            else:
                self._ASSIGNMENT_CACHE[self, other] = assignment.AnatomicalAssignment(self, other, qualification)
        return self._ASSIGNMENT_CACHE[self, other]
