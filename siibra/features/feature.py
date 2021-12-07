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

from ..commons import logger, Registry
from ..core.atlas import Atlas
from ..core.space import Location
from ..core.region import Region
from ..core.parcellation import Parcellation

from typing import Tuple
from abc import ABC, abstractmethod


class Feature(ABC):
    """
    Abstract base class for all data features.
    """

    REGISTRY = Registry()

    def __init__(self):
        self._match = None

    def __init_subclass__(cls):
        """
        Registers all subclasses of Feature.
        """
        logger.debug(
            f"Registering feature type {cls.__name__} with modality {cls.modality()}"
        )
        cls.REGISTRY.add(cls.modality(), cls)

    @property
    def matched(self):
        return self._match is not None

    @property
    def matched_region(self):
        if isinstance(self._match, Region):
            return self._match
        else:
            return None

    @property
    def matched_parcellation(self):
        if isinstance(self._match, Region):
            return self._match.parcellation
        elif isinstance(self._match, Parcellation):
            return self._match
        else:
            return None

    @property
    def matched_location(self):
        if isinstance(self._match, Location):
            return self._match
        else:
            return None

    @abstractmethod
    def match(self, concept):
        """
        Matches this feature to the given atlas concept (or a subconcept of it),
        and remembers the matching result.

        Parameters:
        -----------
        concept : AtlasConcept

        Returns:
        -------
        True, if match was successful, otherwise False
        """
        raise RuntimeError(
            f"matches() needs to be implemented by derived classes of {self.__class__.__name__}"
        )

    def __str__(self):
        return f"{self.__class__.__name__} feature"

    @classmethod
    def modality(cls):
        """Returns a string representing the modality of a feature."""
        return str(cls).split("'")[1].split(".")[-1]


class SpatialFeature(Feature):
    """
    Base class for coordinate-anchored data features.
    """

    def __init__(self, location: Location):
        """
        Initialize a new spatial feature.

        Parameters
        ----------
        location : Location type
            The location, see siibra.core.location
        """
        assert(location is not None)
        Feature.__init__(self)
        self.location = location

    @property
    def space(self):
        return self.location.space

    def match(self, concept):
        """
        Matches this feature to the given atlas concept (or a subconcept of it),
        and remembers the matching result.

        TODO this could use parameters for continuous maps, thresholding and resolution

        Parameters:
        -----------
        concept : AtlasConcept

        Returns:
        -------
        True, if match was successful, otherwise False
        """

        self._match = None
        if self.location is None:
            return False

        if isinstance(concept, Parcellation):
            region = concept.regiontree
            logger.info(f"{self.__class__} matching against root node {region.name} of {concept.name}")
        elif isinstance(concept, Region):
            region = concept
        else:
            logger.warning(f"{self.__class__} cannot match against {concept.__class__} concepts")
            return self.matched

        for tspace in [self.space] + region.supported_spaces:
            if region.defined_in_space(tspace):
                M = region.build_mask(space=tspace)
                if tspace == self.space:
                    if self.location.intersects_mask(M):
                        self._match = region
                    return self.matched
                else:
                    logger.warning(
                        f"{self.__class__.__name__} cannot be tested for {region.name} "
                        f"in {self.space}, testing in {tspace} instead."
                    )
                    location = self.location.warp(tspace)
                    if location.intersects_mask(M):
                        self._match = region
                    return self.matched
        else:
            logger.warning(
                f"Cannot test overlap of {self.location} with {region}"
            )

        return self.matched

    def __str__(self):
        return f"{self.__class__.__name__} at {str(self.location)}"


class RegionalFeature(Feature):
    """
    Base class for region-anchored data features (semantic anchoring to region
    names instead of coordinates).
    TODO store region as an object that has a link to the parcellation
    """

    def __init__(self, regionspec: Tuple[str, Region], species = [], **kwargs):
        """
        Parameters
        ----------
        regionspec : string or Region
            Specifier for the brain region, will be matched at test time
        """
        if not any(map(lambda c: isinstance(regionspec, c), [Region, str])):
            raise TypeError(
                f"invalid type {type(regionspec)} provided as region specification"
            )
        Feature.__init__(self)
        self.regionspec = regionspec
        self.species = species

    @property
    def species_ids(self):
        return [s.get('@id') for s in self.species]

    def match(self, concept):
        """
        Matches this feature to the given atlas concept (or a subconcept of it),
        and remembers the matching result.

        Parameters:
        -----------
        concept : AtlasConcept

        Returns:
        -------
        True, if match was successful, otherwise False
        """

        # first check if any of
        try:
            if isinstance(concept, Region):
                atlases = concept.parcellation.atlases
            if isinstance(concept, Parcellation):
                atlases = concept.atlases
            if isinstance(concept, Atlas):
                atlases = {concept}
            if atlases:
                # if self.species_ids is defined, and the concept is explicitly not in
                # return False
                if all(atlas.species.get('@id') not in self.species_ids for atlas in atlases):
                    return False
        # for backwards compatibility. If any attr is not found, pass
        except AttributeError:
            pass

        self._match = None

        # regionspec might be a specific region, then we can
        # directly test for the object.
        if isinstance(self.regionspec, Region):
            if isinstance(concept, Parcellation):
                return self.regionspec in concept
            elif isinstance(concept, Region):
                return self.regionspec == concept
            elif isinstance(concept, Atlas):
                return any(self.regionspec in p for p in concept.parcellations)

        # otherwise, it is a string and we need to match explicitely.
        spec = self.regionspec.lower()
        if isinstance(concept, Parcellation):
            logger.debug(f"{self.__class__} matching against root node {concept.regiontree.name} of {concept.name}")
            for w in concept.key.split('_'):
                spec = spec.replace(w.lower(), '')
            for match in concept.regiontree.find(spec):
                # TODO what's with the mutation here?
                self._match = match
                return True

        elif isinstance(concept, Region):
            for w in concept.parcellation.key.split('_'):
                if not w.isnumeric() and len(w)>2:
                    spec = spec.replace(w.lower(), '')
            for match in concept.find(spec):
                # TODO what's with the mutation here?
                self._match = match
                return True

        elif isinstance(concept, Atlas):
            logger.debug(
                "Matching regional features against a complete atlas. "
                "This is not efficient and the query may take a while.")
            for w in concept.key.split('_'):
                spec = spec.replace(w.lower(), '')
            for p in concept.parcellations:
                for match in p.regiontree.find(spec):
                    # TODO what's with the mutation here?
                    self._match = match
                    return True
        else:
            logger.warning(f"{self.__class__} cannot match against {concept.__class__} concepts")

        return self.matched

    def __str__(self):
        return f"{self.__class__.__name__} for {self.regionspec}"


class ParcellationFeature(Feature):
    """
    Base class for data features which apply to the atlas as a whole
    instead of a particular location or region. A typical example is a
    connectivity matrix, which applies to all regions in the atlas.
    """

    def __init__(self, parcellationspec):
        """
        Parameters
        ----------
        parcellationspec : str or Parcellation object
            Identifies the underlying parcellation
        """
        Feature.__init__(self)
        self.spec = parcellationspec
        self.parcellations = Parcellation.REGISTRY.find(parcellationspec)

    def match(self, concept):
        """
        Matches this feature to the given atlas concept (or a subconcept of it),
        and remembers the matching result.

        Parameters:
        -----------
        concept : AtlasConcept

        Returns:
        -------
        True, if match was successful, otherwise False
        """
        self._match = None
        if isinstance(concept, Parcellation):
            if concept in self.parcellations:
                self._match = concept
        elif isinstance(concept, Region):
            if concept.parcellation in self.parcellations:
                self._match = concept
        elif isinstance(concept, Atlas):
            logger.debug(
                "Matching a parcellation feature against a complete atlas. "
                "This will return features matching any supported parcellation, "
                "including different parcellation versions.")
            for p in concept.parcellations:
                if p in self.parcellations:
                    self._match = p
                    return True
        else:
            logger.warning(f"{self.__class__} cannot match against {concept.__class__} concepts")

        return self.matched

    def __str__(self):
        return f"{self.__class__.__name__} for {self.spec}"
