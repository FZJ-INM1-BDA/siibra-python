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
"""Handles the relation between study targets and BrainStructures."""

from typing import Callable, Union, List, Dict, Iterable

from ..commons import Species, logger
from ..core.structure import BrainStructure
from ..core.assignment import AnatomicalAssignment, Qualification
from ..locations.location import Location
from ..core.parcellation import Parcellation, find_regions
from ..core.region import Region
from ..core.space import Space
from ..core.concept import get_registry
from ..exceptions import SpaceWarpingFailedError
from ..vocabularies import REGION_ALIASES


class AnatomicalAnchor:
    """
    Anatomical anchor to an atlas region, a geometric primitive in an atlas
    reference space, a reference space, or a combination of these.

    A space anchors a feature to a reference space as a whole: space queries
    match it, while region and location queries only match the anchor's
    region or location.
    """

    _MATCH_MEMO: Dict[str, Dict[Region, Qualification]] = {}

    def __init__(
        self,
        species: Union[List[Species], Species, str],
        location: Union[Location, Callable[[], Location]] = None,
        region: Union[str, Region] = None,
        space: Union[str, Space] = None,
    ):
        """
        Parameters
        ----------
        species : Species, str or list of Species
        location : Location or callable, optional
            The location, or a function without arguments returning it if
            computing it is expensive, e.g. the bounding box of an image. The
            function is called at most once, when the location is first
            needed.
        region : str or Region, optional
        space : str or Space, optional
            The reference space of the anchor, e.g. of a feature's data.
        """

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
        if location is None or isinstance(location, Location):
            self._location_cached, self._location_getter = location, None
        elif callable(location):
            self._location_cached, self._location_getter = None, location
        else:
            raise ValueError(f"Invalid location specification: {location}")
        self._location_deferred = self._location_getter is not None
        self._space_spec = space
        self._space_cached = space if isinstance(space, Space) else None
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
        if self._location_getter is not None:
            self._location_cached = self._location_getter()
            self._location_getter = None
        return self._location_cached

    @property
    def _has_location(self) -> bool:
        """Whether the anchor has a location, without computing a deferred one."""
        return self._location_cached is not None or self._location_getter is not None

    @property
    def parcellations(self) -> List[Parcellation]:
        """
        Return any parcellation objects that regions of this anchor belong to.
        """
        return list({region.root for region in self.regions})

    @property
    def space(self) -> Space:
        """The reference space of the anchor: its own, or else its location's."""
        if self._space_spec is not None:
            if self._space_cached is None:
                self._space_cached = get_registry("Space").get(self._space_spec)
            return self._space_cached
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
                for region in find_regions(self._regionspec, filter_children=True, find_topmost=False)
                if region.species in self.species
            }
            # add more regions from possible aliases of the region spec
            for alt_species, aliases in self.region_aliases.items():
                for alias_regionspec, qualificationspec in aliases.items():
                    for r in find_regions(alias_regionspec, filter_children=True, find_topmost=False):
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
        if self._space_spec is not None and (self._location_deferred or not self._has_location):
            # described by the space, so that a deferred location is never
            # computed for it, and the text does not depend on whether it was
            return " ".join(filter(None, [region, f"in {self.space.name}"]))
        location = str(self.location) if self._has_location else ""
        separator = " " if min(len(region), len(location)) > 0 else ""
        if region and location:
            return region + " with " + location
        else:
            return region + separator + location

    def assign(self, concept: Union[BrainStructure, Space]) -> AnatomicalAssignment:
        """
        Match this anchor to a query concept. Assignments are cached at runtime,
        so repeated assignment with the same concept will be cheap.
        """
        if isinstance(concept, Space):
            # an anchor's own space decides, without needing its location
            if self._space_spec is not None:
                if self.space.matches(concept):
                    return [AnatomicalAssignment(concept, self.space, Qualification.CONTAINED)]
                return []
            if self._has_location and self.location.space.matches(concept):
                return [AnatomicalAssignment(concept, self.location, Qualification.CONTAINED)]
            else:
                return []

        if concept not in self._assignments:
            assignments: List[AnatomicalAssignment] = []

            skip_location_assignment = False
            if (
                len(self.regions) > 0
                and self._has_location
                and isinstance(concept, Region)
            ):
                skip_location_assignment = True

            for region in self.regions:
                region_assignment = region.assign(concept)
                if region_assignment is not None:
                    skip_location_assignment = True
                    assignments.append(region_assignment)

            if self._has_location and not skip_location_assignment:
                try:
                    assignments.append(self.location.assign(concept))
                except SpaceWarpingFailedError as e:
                    logger.debug(e)

            self._assignments[concept] = sorted(a for a in assignments if a is not None)

        self._last_matched_concept = concept if len(self._assignments[concept]) > 0 else None
        return self._assignments[concept]

    def matches(self, concept: Union[BrainStructure, Space]) -> bool:
        return len(self.assign(concept)) > 0

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

        both_spaces = self._space_spec is not None and other._space_spec is not None
        space = self.space if both_spaces and self.space == other.space else None

        return AnatomicalAnchor(species, location, regions, space=space)

    def __radd__(self, other) -> 'AnatomicalAnchor':
        # required to enable `sum`
        return self if other == 0 else self.__add__(other)
