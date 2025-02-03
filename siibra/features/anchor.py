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
"""Handles the relation between study targets and AtlasConcepts."""

from ..commons import logger, Species

from ..core.concept import AtlasConcept
from ..locations.location import Location
from ..locations.boundingbox import BoundingBox
from ..core.parcellation import Parcellation
from ..core.region import Region
from ..core.space import Space
from ..core.relation_qualification import Qualification as AssignmentQualification, RelationAssignment

from ..vocabularies import REGION_ALIASES

from typing import Union, List, Dict

AnatomicalAssignment = RelationAssignment[Union[Region, Location]]


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
            self._regionspec = region
        else:
            if region is not None:
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
        if self.location is None:
            return None
        else:
            return self.location.space

    @property
    def region_aliases(self):
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
        # decoding region strings is quite compute intensive, so we cache this at the class level
        if self._regions_cached is not None:
            return self._regions_cached

        if self._regionspec is None:
            self._regions_cached = {}
            return self._regions_cached

        if self._regionspec not in self.__class__._MATCH_MEMO:
            self._regions_cached = {}
            # decode the region specification into a set of region objects
            regions = {
                r: AssignmentQualification.EXACT
                for species in self.species
                for r in Parcellation.find_regions(self._regionspec)
                if r.species == species
            }
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

    def __repr__(self):
        return self.__str__()

    def assign(self, concept: AtlasConcept):
        """
        Match this anchoring to an atlas concept.
        """
        if concept not in self._assignments:
            matches: List[AnatomicalAssignment] = []
            if isinstance(concept, Space) and self.space == concept:
                matches.append(
                    AnatomicalAssignment(self.space, concept, AssignmentQualification.EXACT)
                )
            elif isinstance(concept, Region) and concept.species in self.species:
                hierarchy_search = concept.root.find(self._regionspec)
                if len(hierarchy_search) > 0 or self.has_region_aliases:  # dramatic speedup, since decoding _regionspec is expensive
                    for r in self.regions:
                        matches.append(AnatomicalAnchor.match_regions(r, concept))
                if len(hierarchy_search) == 0 and self.location is not None:
                    # We perform the (quite expensive) location-to-region test
                    # only if this anchor's regionspec is not known to the
                    # parcellation of the query region. Otherwise we can rely
                    # on the region-to-region test.
                    matches.append(AnatomicalAnchor.match_location_to_region(self.location, concept))
            elif isinstance(concept, Location):
                if self.location is not None:
                    matches.append(AnatomicalAnchor.match_locations(self.location, concept))
                for region in self.regions:
                    match = AnatomicalAnchor.match_location_to_region(concept, region)
                    matches.append(None if match is None else match.invert())
            self._assignments[concept] = sorted(m for m in matches if m is not None)

        self._last_matched_concept = concept \
            if len(self._assignments[concept]) > 0 \
            else None

        return self._assignments[concept]

    def matches(self, concept: AtlasConcept):
        return len(self.assign(concept)) > 0

    @classmethod
    def match_locations(cls, location1: Location, location2: Location):
        assert all(isinstance(loc, Location) for loc in [location1, location2])
        if (location1, location2) not in cls._MATCH_MEMO:
            if location1 == location2:
                res = AnatomicalAssignment(location1, location2, AssignmentQualification.EXACT)
            elif location1.contained_in(location2):
                res = AnatomicalAssignment(location1, location2, AssignmentQualification.CONTAINED)
            elif location1.contains(location2):
                res = AnatomicalAssignment(location1, location2, AssignmentQualification.CONTAINS)
            elif location1.intersects(location2):
                res = AnatomicalAssignment(location1, location2, AssignmentQualification.OVERLAPS)
            else:
                res = None
            cls._MATCH_MEMO[location1, location2] = res
        return cls._MATCH_MEMO[location1, location2]

    @classmethod
    def match_regions(cls, region1: Region, region2: Region):
        assert all(isinstance(r, Region) for r in [region1, region2])
        if (region1, region2) not in cls._MATCH_MEMO:
            if region1 == region2:
                res = AnatomicalAssignment(region1, region2, AssignmentQualification.EXACT)
            elif region1 in region2:
                res = AnatomicalAssignment(region1, region2, AssignmentQualification.CONTAINED)
            elif region2 in region1:
                res = AnatomicalAssignment(region1, region2, AssignmentQualification.CONTAINS)
            else:
                res = None
            cls._MATCH_MEMO[region1, region2] = res
        return cls._MATCH_MEMO[region1, region2]

    @classmethod
    def match_location_to_region(cls, location: Location, region: Region):
        assert isinstance(location, Location)
        assert isinstance(region, Region)

        for subregion in region.children:
            res = cls.match_location_to_region(location, subregion)
            if res is not None:
                return res

        if (location, region) not in cls._MATCH_MEMO:
            # compute mask of the region
            mask = None
            if region.mapped_in_space(location.space, recurse=False):
                if (region, location.space) not in cls._MASK_MEMO:
                    cls._MASK_MEMO[region, location.space] = \
                        region.fetch_regional_map(space=location.space, maptype='labelled')
                mask = cls._MASK_MEMO[region, location.space]
                mask_space = location.space
                expl = (
                    f"{location} was compared with the mask of query region "
                    f"'{region.name}' in {location.space.name}."
                )
            else:
                for space in region.supported_spaces:
                    if not region.mapped_in_space(space, recurse=False):
                        continue
                    if not space.provides_image:  # siibra does not yet match locations to surface spaces
                        continue
                    if (region, space) not in cls._MASK_MEMO:
                        cls._MASK_MEMO[region, space] = region.fetch_regional_map(space=space, maptype='labelled')
                    mask = cls._MASK_MEMO[region, space]
                    mask_space = space
                    if location.space == mask_space:
                        expl = (
                            f"{location} was compared with the mask of query region '{region.name}' "
                            f"in {mask_space}."
                        )
                    else:
                        expl = (
                            f"{location} was warped from {location.space.name} and then compared "
                            f"in this space with the mask of query region '{region.name}'."
                        )
                    if mask is not None:
                        break

            if mask is None:
                logger.debug(
                    f"'{region.name}' provides no mask in a space "
                    f"to which {location} can be warped."
                )
                res = None
            else:
                # compare mask to location
                loc_warped = location.warp(mask_space)
                if loc_warped is None:
                    # seems we cannot warp our location to the mask space
                    # this typically happens when the location extends outside
                    # the brain. We might still be able the warp the
                    # bounding box of the mask to the location and check.
                    # TODO in fact we should estimate an affine matrix from the warped bounding box,
                    # and resample the mask for the inverse test to be more precise.
                    bbox_mask = BoundingBox.from_image(mask, mask_space).warp(location.space)
                    return cls.match_locations(location, bbox_mask)
                elif loc_warped.contained_in(mask):
                    res = AnatomicalAssignment(location, region, AssignmentQualification.CONTAINED, expl)
                elif loc_warped.contains(mask):
                    res = AnatomicalAssignment(location, region, AssignmentQualification.CONTAINS, expl)
                elif loc_warped.intersects(mask):
                    res = AnatomicalAssignment(location, region, AssignmentQualification.OVERLAPS, expl)
                else:
                    logger.debug(
                        f"{location} does not match mask of '{region.name}' in {mask_space.name}."
                    )
                    res = None
            cls._MATCH_MEMO[location, region] = res

        # keep mask cache small
        if len(cls._MASK_MEMO) > 3:
            # from Python 3.6, this remove the *oldest* entry
            cls._MASK_MEMO.pop(next(iter(cls._MASK_MEMO)))

        return cls._MATCH_MEMO[location, region]

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
