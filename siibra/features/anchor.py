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

from .. import logger, find_regions

from ..core.concept import AtlasConcept
from ..core.location import Location
from ..core.parcellation import Parcellation
from ..core.region import Region

from ..vocabularies import REGION_ALIASES

from typing import Union
from enum import Enum


class AssignmentQualification(Enum):
    EXACT = 1
    OVERLAPS = 2
    CONTAINED = 3
    CONTAINS = 4

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
        }
        return transl[self.name]

    def invert(self):
        """
        Return a MatchPrecision object with the inverse meaning
        """
        inverses = {
            "EXACT": "EXACT",
            "OVERLAPS": "OVERLAPS",
            "CONTAINED": "CONTAINS",
            "CONTAINS": "CONTAINED",
        }
        return AssignmentQualification[inverses[self.name]]


class ListWithoutNone(list):
    """ A list which ignores None-type elements during append()."""

    def __init__(self, **kwargs):
        list.__init__(self, **kwargs)

    def append(self, element):
        if element is None:
            return
        list.append(self, element)


class AnatomicalAssignment:
    """
    An assignment of an anatomical anchor to an atlas concept.
    """

    def __init__(
        self,
        assigned_structure: Union[Region, Location],
        qualification: AssignmentQualification,
        explanation: str = ""
    ):
        self.assigned_structure = assigned_structure
        self.qualification = qualification
        self.explanation = explanation.strip()

    @property
    def is_exact(self):
        return self.qualification == AssignmentQualification.EXACT

    def __str__(self):
        msg = f"Element {self.qualification.verb} '{self.assigned_structure}'."
        return msg if self.explanation == "" else f"{msg} ({self.explanation})."

    def invert(self):
        return AnatomicalAssignment(self.assigned_structure, self.qualification.invert(), self.explanation)

    def __lt__(self, other):
        return self.qualification.value < other.assignment_type.value


class AnatomicalAnchor:
    """
    Anatomical anchor to an atlas region,
    a geometric primitive in an atlas reference space,
    or both.
    """

    _MATCH_MEMO = {}

    def __init__(self, location: Location = None, region: Union[str, Region] = None, species: str = None):

        if not any(s is not None for s in [location, region]):
            raise ValueError(
                "To define a localization, a region and/or "
                "location needs to be specified."
            )
        self.location = location
        self.species = species
        self._assignments = {}
        if isinstance(region, Region):
            self._regions_cached = [region]
            self._regionspec = None
        else:
            assert isinstance(region, str)
            self._regions_cached = None
            self._regionspec = region

    @property
    def regions(self):
        if self._regions_cached is None:
            # decode the region specification into a set of region objects
            self._regions_cached = {
                r: AssignmentQualification['EXACT'] 
                for r in find_regions(self._regionspec, self.species)
            }
            # add more regions from possible aliases of the region spec
            region_aliases = REGION_ALIASES.get(self.species, {}).get(self._regionspec, {})
            for species, aliases in region_aliases.items():
                for regionspec, qualificationspec in aliases.items():
                    for r in find_regions(regionspec, species):
                        if r not in self._regions_cached:
                            logger.info(f"Adding region {r.name} in {species} from alias to {self._regionspec}")
                            self._regions_cached[r] = qualificationspec

        return self._regions_cached

    def __str__(self):
        region = "" if self._regionspec is None else str(self._regionspec)
        location = "" if self.location is None else str(self.location)
        separator = " " if min(len(region), len(location)) > 0 else ""
        return region + separator + location

    def assign(self, concept: AtlasConcept):
        """
        Match this anchoring to an atlas concept.
        """
        if concept not in self._assignments:
            matches = ListWithoutNone()
            if isinstance(concept, Region):
                if self.location is not None:
                    logger.info(f"match location {self.location} to region '{concept.name}'")
                    matches.append(AnatomicalAnchor.match_location_to_region(self.location, concept))
                for region in self.regions:
                    matches.append(AnatomicalAnchor.match_regions(region, concept))
            elif isinstance(concept, Location):
                if self.location is not None:
                    matches.append(AnatomicalAnchor.match_locations(self.location, concept))
                for region in self.regions:
                    match = AnatomicalAnchor.match_location_to_region(concept, region)
                    matches.append(None if match is None else match.invert())
            self._assignments[concept] = sorted(matches)
        return self._assignments[concept]

    def matches(self, concept: AtlasConcept):
        return len(self.assign(concept)) > 0

    @classmethod
    def match_locations(cls, location1: Location, location2: Location):
        assert all(isinstance(loc, Location) for loc in [location1, location2])
        if (location1, location2) not in cls._MATCH_MEMO:
            if location1 == location2:
                res = AnatomicalAssignment(location2, AssignmentQualification.EXACT)
            elif location1.contained_in(location2):
                res = AnatomicalAssignment(location2, AssignmentQualification.CONTAINED)
            elif location1.contains(location2):
                res = AnatomicalAssignment(location2, AssignmentQualification.CONTAINS)
            elif location1.intersects(location2):
                res = AnatomicalAssignment(location2, AssignmentQualification.OVERLAPS)
            else:
                res = None
            cls._MATCH_MEMO[location1, location2] = res
        return cls._MATCH_MEMO[location1, location2]

    @classmethod
    def match_regions(cls, region1: Region, region2: Region):
        assert all(isinstance(r, Region) for r in [region1, region2])
        if (region1, region2) not in cls._MATCH_MEMO:
            logger.debug(f"match region {region1} to region '{region2}'")
            if region1 == region2:
                res = AnatomicalAssignment(region2, AssignmentQualification.EXACT)
            elif region1 in region2:
                res = AnatomicalAssignment(region2, AssignmentQualification.CONTAINED)
            elif region2 in region1:
                res = AnatomicalAssignment(region2, AssignmentQualification.CONTAINS)
            else:
                res = None
            cls._MATCH_MEMO[region1, region2] = res
        return cls._MATCH_MEMO[region1, region2]

    @classmethod
    def match_location_to_region(cls, location: Location, region: Region):
        assert isinstance(location, Location)
        assert isinstance(region, Region)
        if (location, region) not in cls._MATCH_MEMO:
            mask = region.build_mask(space=location.space, maptype='labelled')
            expl = (
                f"{location} was compared with the mask of query region "
                f"'{region.name}' in {location.space.name}."
            )
            if mask is None:
                logger.warn(f"'{region.name}' provides not mask to match {location}.")
                res = None
            elif location.contained_in(mask):
                res = AnatomicalAssignment(region, AssignmentQualification.CONTAINED, expl)
            elif location.contains(mask):
                res = AnatomicalAssignment(region, AssignmentQualification.CONTAINS, expl)
            elif location.intersects(mask):
                res = AnatomicalAssignment(region, AssignmentQualification.OVERLAPS, expl)
            else:
                logger.debug(
                    f"{location} does not match mask of '{region.name}' "
                    f"in {location.space.name}."
                )
                res = None
            cls._MATCH_MEMO[location, region] = res
        return cls._MATCH_MEMO[location, region]

    def represented_parcellations(self):
        """
        Return any parcellation objects that this anchor explicitly points to.
        """
        return [
            r for r in self.regions
            if isinstance(r, Parcellation)
        ]
