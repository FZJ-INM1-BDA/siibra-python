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

from ..core.region import Region
from .location import Location

from enum import Enum
from typing import Union


class AssignmentQualification(Enum):
    EXACT = 1
    OVERLAPS = 2
    CONTAINED = 3
    CONTAINS = 4
    APPROXIMATE = 5

    @property
    def verb(self):
        """
        a string that can be used as a verb in a sentence
        for producing human-readable messages.
        """
        transl = {
            'EXACT': 'coincides with',
            'OVERLAPS': 'overlaps with',
            'CONTAINED': 'is contained in',
            'CONTAINS': 'contains',
            'APPROXIMATE': 'approximates to',
        }
        return transl[self.name]

    def invert(self):
        """
        Return qualification with the inverse meaning
        """
        inverses = {
            "EXACT": "EXACT",
            "OVERLAPS": "OVERLAPS",
            "CONTAINED": "CONTAINS",
            "CONTAINS": "CONTAINED",
            "APPROXIMATE": "APPROXIMATE",
        }
        return AssignmentQualification[inverses[self.name]]

    def __str__(self):
        return f"{self.__class__.__name__}={self.name.lower()}"

    def __repr__(self):
        return str(self)


class AnatomicalAssignment:
    """
    Represents a qualified assignment between anatomical structures.
    """

    def __init__(
        self,
        query_structure: Union[Region, Location],
        assigned_structure: Union[Region, Location],
        qualification: AssignmentQualification,
        explanation: str = ""
    ):
        self.query_structure = query_structure
        self.assigned_structure = assigned_structure
        self.qualification = qualification
        self.explanation = explanation.strip()

    @property
    def is_exact(self):
        return self.qualification == AssignmentQualification.EXACT

    def __str__(self):
        msg = f"'{self.query_structure}' {self.qualification.verb} '{self.assigned_structure}'"
        return msg if self.explanation == "" else f"{msg} - {self.explanation}"

    def invert(self):
        return AnatomicalAssignment(self.assigned_structure, self.query_structure, self.qualification.invert(), self.explanation)

    def __lt__(self, other: 'AnatomicalAssignment'):
        if not isinstance(other, AnatomicalAssignment):
            raise ValueError(f"Cannot compare AnatomicalAssignment with instances of '{type(other)}'")
        return self.qualification.value < other.qualification.value
