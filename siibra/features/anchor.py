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
"""Handles the relation between study targets and BrainStructures."""

from ..commons import Species, logger

from ..core.structure import BrainStructure
from ..core.assignment import AnatomicalAssignment, Qualification
from ..locations.location import Location
from ..core.parcellation import Parcellation
from ..core.region import Region
from ..core.space import Space
from ..exceptions import SpaceWarpingFailedError

from ..vocabularies import REGION_ALIASES

from typing import Union, List, Dict, Iterable


class AnatomicalAnchor:
    """
    Anatomical anchor to an atlas region, a geometric primitive in an atlas
    reference space, or both.
    """

    _MATCH_MEMO: Dict[str, Dict[Region, Qualification]] = {}

    def __init__(
        self,
        species: Union[List[Species], Species, str],
        location: Location = None,
        region: Union[str, Region] = None
    ):

        if isinstance(species, (str, Species)):
            self.species = {Species.decode(species)}
        elif isinstance(species, Iterable):
            assert all(isinstance(_, Species) for _ in species)
            self.species = set(species)
        else:
            sp = Species.decode(species)
            if sp is None:
                raise ValueError(f"Invalid species specification: {species}")
            else:
                self.species = {sp}
        self._location_cached = location
        self._assignments: Dict[BrainStructure, List[AnatomicalAssignment]] = {}
        self._last_matched_concept = None
        if isinstance(region, dict):
            self._regions_cached = region
            self._regionspec = ", ".join({r.name for r in region.keys()})
        else:
            self._regions_cached = None
            self._regionspec = None
            if isinstance(region, Region):
                self._regions_cached = {region: Qualification.EXACT}
            elif isinstance(region, str):
                # we will decode regions only when needed, see self.regions property
                self._regionspec = region
            else:
                if region is not None:
                    raise ValueError(f"Invalid region specification: {region}")
        self._aliases_cached = None

    @property
    def location(self) -> Location:
        # allow to overwrite in derived classes
        return self._location_cached

    @property
    def parcellations(self) -> List[Parcellation]:
        """
        Return any parcellation objects that regions of this anchor belong to.
        """
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
    def has_region_aliases(self) -> bool:
        return len(self.region_aliases) > 0

    @property
    def regions(self) -> Dict[Region, Qualification]:
        """
        Return the list of regions associated with this anchor.
        Decode the self._regionspec string into region objects now,
        if applicable and called for the first time.
        """
        # decoding region strings is quite compute intensive, so we cache this at the class level
        if self._regions_cached is not None:
            return self._regions_cached

        if self._regionspec is None:
            self._regions_cached = dict()
            return self._regions_cached

        match_key = self._regionspec + '-' + str(self.species)
        if match_key not in self.__class__._MATCH_MEMO:
            # decode the region specification into a dict of region objects and assignment qualifications
            regions = {
                region: Qualification.EXACT
                for region in Parcellation.find_regions(self._regionspec)
                if region.species in self.species
            }
            # add more regions from possible aliases of the region spec
            for alt_species, aliases in self.region_aliases.items():
                for alias_regionspec, qualificationspec in aliases.items():
                    for r in Parcellation.find_regions(alias_regionspec):
                        if r.species != alt_species:
                            continue
                        if r not in regions:
                            regions[r] = Qualification[qualificationspec.upper()]

            self.__class__._MATCH_MEMO[match_key] = regions
        self._regions_cached = self.__class__._MATCH_MEMO[match_key]

        return self._regions_cached

    def __str__(self):
        parcs = {p.id: p.name for p in self.represented_parcellations()}
        if len(parcs) == 1 and self._regionspec in [pid for pid in parcs]:
            region = parcs[self._regionspec]  # if parcellation was anchored with the id instead of the name
        else:
            region = "" if self._regionspec is None else str(self._regionspec)
        location = "" if self.location is None else str(self.location)
        separator = " " if min(len(region), len(location)) > 0 else ""
        if region and location:
            return region + " with " + location
        else:
            return region + separator + location

    def assign(self, concept: BrainStructure, restrict_space: bool = False) -> AnatomicalAssignment:
        """
        Match this anchor to a query concept. Assignments are cached at runtime,
        so repeated assignment with the same concept will be cheap.
        """
        if (
            restrict_space
            and self.location is not None
            and isinstance(concept, Location)
            and not self.location.space.matches(concept.space)
        ):
            return []
        if concept not in self._assignments:
            assignments: List[AnatomicalAssignment] = []
            if self.location is not None:
                try:
                    assignments.append(self.location.assign(concept))
                except SpaceWarpingFailedError as e:
                    logger.debug(e)
            for region in self.regions:
                assignments.append(region.assign(concept))
            self._assignments[concept] = sorted(a for a in assignments if a is not None)

        self._last_matched_concept = concept \
            if len(self._assignments[concept]) > 0 \
            else None
        return self._assignments[concept]

    def matches(self, concept: BrainStructure, restrict_space: bool = False) -> bool:
        return len(self.assign(concept, restrict_space)) > 0

    def represented_parcellations(self) -> List[Parcellation]:
        """
        Return any parcellation objects that this anchor explicitly points to.
        """
        return [r for r in self.regions if isinstance(r, Parcellation)]

    @property
    def last_match_result(self) -> List[AnatomicalAssignment]:
        return self._assignments.get(self._last_matched_concept, [])

    @property
    def last_match_description(self) -> str:
        if self.last_match_result is None:
            return ""
        else:
            return ' and '.join({str(_) for _ in self.last_match_result})

    def __add__(self, other: 'AnatomicalAnchor') -> 'AnatomicalAnchor':
        if not isinstance(other, AnatomicalAnchor):
            raise ValueError(f"Cannot combine an AnatomicalAnchor with {other.__class__}")

        if self.species != other.species:
            raise ValueError("Cannot combine an AnatomicalAnchor from different species.")
        else:
            species = self.species.union(other.species)

        regions = self.regions
        regions.update(other.regions)

        location = Location.union(self.location, other.location)

        return AnatomicalAnchor(species, location, regions)

    def __radd__(self, other) -> 'AnatomicalAnchor':
        # required to enable `sum`
        return self if other == 0 else self.__add__(other)
