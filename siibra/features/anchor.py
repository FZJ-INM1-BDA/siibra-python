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

from ..commons import Species

from ..core.concept import AtlasConcept
from ..core.assignment import AnatomicalAssignment, AssignmentQualification
from ..locations.location import Location
from ..core.parcellation import Parcellation
from ..core.region import Region
from ..core.space import Space

from ..vocabularies import REGION_ALIASES

from typing import Union, List, Dict


class AnatomicalAnchor:
    """
    Anatomical anchor to an atlas region,
    a geometric primitive in an atlas reference space,
    or both.
    """

    _MATCH_MEMO: Dict[str, Dict[Region, AssignmentQualification]] = {}
    _MASK_MEMO = {}

    def __init__(self, species: Union[List[Species], Species, str], location: Location = None, region: Union[str, Region] = None):

        if isinstance(species, (str, Species)):
            self.species = {Species.decode(species)}
        elif isinstance(species, list):
            assert all(isinstance(_, Species) for _ in species)
            self.species = set(species)
        else:
            sp = Species.decode(species)
            if sp is None:
                raise ValueError(f"Invalid species specification: {species}")
            else:
                self.species = {sp}
        self._location_cached = location
        self._assignments: Dict[Union[AtlasConcept, Location], List[AnatomicalAssignment]] = {}
        self._last_matched_concept = None
        self._regions_cached = None
        self._regionspec = None

        if isinstance(region, Region):
            self._regions_cached = {region: AssignmentQualification.EXACT}
        elif isinstance(region, str):
            # we will decode regions only when needed, see self.regions property
            self._regionspec = region
        elif region is not None:
            raise ValueError(f"Invalid region specification: {region}")
        self._aliases_cached = None

    @property
    def location(self):
        # allow to overwrite in derived classes
        return self._location_cached

    @property
    def parcellations(self) -> List[Parcellation]:
        return list({region.root for region in self.regions})

    @property
    def space(self) -> Space:
        # may be overriden by derived classes, e.g. in features.VolumeOfInterest
        return None if self.location is None else self.location.space

    @property
    def region_aliases(self):
        # return any predefined aliases for the region specified in this anchor.
        if self._aliases_cached is None:
            self._aliases_cached: Dict[str, Dict[str, str]] = {
                Species.decode(species_str): region_alias_mapping
                for s in self.species
                for species_str, region_alias_mapping in REGION_ALIASES.get(str(s), {}).get(self._regionspec, {}).items()
            }
        return self._aliases_cached

    @property
    def has_region_aliases(self):
        return len(self.region_aliases) > 0

    @property
    def regions(self) -> Dict[Region, AssignmentQualification]:
        """
        Return the list of regions associated with this anchor.
        Decode the self._regionspec string into region objects now,
        if applicable and called for the first time.
        """
        # decoding region strings is quite compute intensive, so we cache this at the class level
        if self._regions_cached is not None:
            return self._regions_cached

        elif self._regionspec is None:
            self._regions_cached = {}
            return self._regions_cached

        elif self._regionspec not in self.__class__._MATCH_MEMO:
            self._regions_cached = {}
            # decode the region specification into a set of region objects
            regions = dict()
            for p in Parcellation.registry():
                if p.species not in self.species:
                    continue
                try:
                    regions[p.get_region(self._regionspec)] = AssignmentQualification.EXACT
                except Exception:
                    pass
            # add more regions from possible aliases of the region spec
            for alt_species, aliases in self.region_aliases.items():
                for regionspec, qualificationspec in aliases.items():
                    for r in Parcellation.find_regions(regionspec):
                        if r.species != alt_species:
                            continue
                        if r not in self._regions_cached:
                            regions[r] = AssignmentQualification[qualificationspec.upper()]

            self.__class__._MATCH_MEMO[self._regionspec] = regions
        self._regions_cached = self.__class__._MATCH_MEMO[self._regionspec]

        return self._regions_cached

    def __str__(self):
        region = "" if self._regionspec is None else str(self._regionspec)
        location = "" if self.location is None else str(self.location)
        separator = " " if min(len(region), len(location)) > 0 else ""
        return region + separator + location

    def assign(self, concept: AtlasConcept):
        """
        Match this anchor to a query concept.
        Assignments are cached at runtime,
        so repeated assignment with the same concept will be cheap.
        """
        if concept not in self._assignments:
            matches: List[AnatomicalAssignment] = []
            if self.location is not None:
                matches.append(self.location.assign(concept))
            for region in self.regions:
                matches.append(region.assign(concept))

            self._assignments[concept] = sorted(m for m in matches if m is not None)

        self._last_matched_concept = concept \
            if len(self._assignments[concept]) > 0 \
            else None
        return self._assignments[concept]

    def matches(self, concept: AtlasConcept):
        return len(self.assign(concept)) > 0

    def represented_parcellations(self):
        """
        Return any parcellation objects that this anchor explicitly points to.
        """
        return [
            r for r in self.regions
            if isinstance(r, Parcellation)
        ]

    @property
    def last_match_result(self) -> List[AnatomicalAssignment]:
        return self._assignments.get(self._last_matched_concept, [])

    @property
    def last_match_description(self) -> str:
        if self.last_match_result is None:
            return ""
        else:
            return ' and '.join({str(_) for _ in self.last_match_result})
