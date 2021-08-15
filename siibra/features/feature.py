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
        pass

    def __init_subclass__(cls):
        """
        Registers all subclasses of Feature.
        """
        logger.debug(
            f"Registering feature type {cls.__name__} with modality {cls.modality()}"
        )
        cls.REGISTRY.add(cls.modality(), cls)

    @abstractmethod
    def matches(self, concept):
        """
        Returns True if this feature should be considered part
        of the given atlas concept, otherwise else.

        Parameters:
        -----------
        concept : AtlasConcept
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

    def matches(self, concept: Region):
        """
        Returns true if the location information of this feature overlaps
        with the provided atlas concept.

        Parameters:
        -----------
        region : Region
        """
        if self.location is None:
            return False

        if isinstance(concept, Parcellation):
            region = concept.regiontree
            logger.info(f"{self.__class__} matching against root node {region.name} of {concept.name}")
        elif isinstance(concept, Region):
            region = concept
        else:
            logger.warning(f"{self.__class__} cannot match against {concept.__class__} concepts")
            return False

        for tspace in [self.space] + region.supported_spaces:
            if region.defined_in_space(tspace):
                M = region.build_mask(space=tspace)
                if tspace == self.space:
                    return self.location.intersects_mask(M)
                else:
                    logger.warning(
                        f"{self.__class__.__name__} cannot be tested for {region.name} in {self.space}, testing in {tspace} instead."
                    )
                    return self.location.warp(tspace).intersects_mask(M)
        else:
            logger.warning(
                f"Cannot test overlap of {self.location} with {region}"
            )
            return False

    def __str__(self):
        return f"{self.__class__.__name__} at {str(self.location)}"


class RegionalFeature(Feature):
    """
    Base class for region-anchored data features (semantic anchoring to region
    names instead of coordinates).
    TODO store region as an object that has a link to the parcellation
    """

    def __init__(self, regionspec: Tuple[str, Region]):
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

    def matches(self, concept: Region):
        """
        Returns true if this feature is linked to the given region.

        Parameters:
        -----------
        concept : Region
        """
        if isinstance(concept, Parcellation):
            logger.debug(f"{self.__class__} matching against root node {concept.regiontree.name} of {concept.name}")
            return len(concept.regiontree.find(self.regionspec)) > 0
        elif isinstance(concept, Region):
            return len(concept.find(self.regionspec)) > 0
        elif isinstance(concept, Atlas):
            logger.debug(
                "Matching regional features against a complete atlas. "
                "This is not efficient and the query may take a while.")
            return any(
                len(p.regiontree.find(self.regionspec)) > 0
                for p in concept.parcellations)
        else:
            logger.warning(f"{self.__class__} cannot match against {concept.__class__} concepts")
            return False        


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

    def matches(self, concept: Parcellation):
        """
        Returns true if this global feature is related to the given atlas selection.

        Parameters:
        -----------
        concept : Parcellation
        """
        if isinstance(concept, Parcellation):
            return concept in self.parcellations
        elif isinstance(concept, Region):
            return concept.parcellation in self.parcellations
        elif isinstance(concept, Atlas):
            logger.debug(
                "Matching a parcellation feature against a complete atlas. "
                "This will return features matching any supported parcellation, "
                "including different parcellation versions.")
            return any(p in self.parcellations for p in concept.parcellations)
        else:
            logger.warning(f"{self.__class__} cannot match against {concept.__class__} concepts")
            return False


    def __str__(self):
        return f"{self.__class__.__name__} for {self.spec}"
