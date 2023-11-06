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
"""Qualification between two arbitary concepts"""

from enum import Enum
from typing import Dict, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar("T")


class Qualification(Enum):
    EXACT = 1
    OVERLAPS = 2
    CONTAINED = 3
    CONTAINS = 4
    APPROXIMATE = 5
    HOMOLOGOUS = 6
    OTHER_VERSION = 7

    @property
    def verb(self):
        """
        a string that can be used as a verb in a sentence
        for producing human-readable messages.
        """
        transl = {
            Qualification.EXACT: 'coincides with',
            Qualification.OVERLAPS: 'overlaps with',
            Qualification.CONTAINED: 'is contained in',
            Qualification.CONTAINS: 'contains',
            Qualification.APPROXIMATE: 'approximates to',
            Qualification.HOMOLOGOUS: 'is homologous to',
            Qualification.OTHER_VERSION: 'is another version of',
        }
        assert self in transl, f"{str(self)} verb cannot be found!"
        return transl[self]

    def invert(self):
        """
        Return a MatchPrecision object with the inverse meaning
        """
        inverses = {
            Qualification.EXACT: Qualification.EXACT,
            Qualification.OVERLAPS: Qualification.OVERLAPS,
            Qualification.CONTAINED: Qualification.CONTAINS,
            Qualification.CONTAINS: Qualification.CONTAINED,
            Qualification.APPROXIMATE: Qualification.APPROXIMATE,
            Qualification.HOMOLOGOUS: Qualification.HOMOLOGOUS,
            Qualification.OTHER_VERSION: Qualification.OTHER_VERSION,
        }
        assert self in inverses, f"{str(self)} inverses cannot be found"
        return inverses[self]

    def __str__(self):
        return f"{self.__class__.__name__}={self.name.lower()}"

    def __repr__(self):
        return str(self)

    @staticmethod
    def parse_relation_assessment(spec: Dict):
        name = spec.get("name")
        if name == "is homologous to":
            return Qualification.HOMOLOGOUS
        raise Exception(f"Cannot parse spec: {spec}")


@dataclass
class RelationAssignment(Generic[T]):
    query_structure: T
    assigned_structure: T
    qualification: Qualification
    explanation: str = ""

    @property
    def is_exact(self):
        return self.qualification == Qualification.EXACT

    def __str__(self):
        msg = f"'{self.query_structure}' {self.qualification.verb} '{self.assigned_structure}'"
        return msg if self.explanation == "" else f"{msg} - {self.explanation}"

    def invert(self):
        return RelationAssignment(
            self.assigned_structure,
            self.query_structure,
            self.qualification.invert(),
            self.explanation
        )

    def __lt__(self, other: 'RelationAssignment'):
        if not isinstance(other, RelationAssignment):
            raise ValueError(f"Cannot compare RelationAssignment with instances of '{type(other)}'")
        return self.qualification.value < other.qualification.value
