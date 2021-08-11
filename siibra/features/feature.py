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
    def matches(self, selection):
        """
        Returns True if this feature should be considered part
        of the given atlas selection, otherwise else.

        Parameters:
        -----------
        selection : AtlasSelection
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
        Feature.__init__(self)
        self.location = location

    def matches(self, selection):
        """
        Returns true if the location information of this feature overlaps
        with the given atlas selection, according to the mask computed in
        the reference space.

        Parameters:
        -----------
        selection : AtlasSelection
        """
        if self.location is None:
            return False

        atlas = selection._atlas
        for tspace in [self.space] + atlas.spaces:
            if selection.region.defined_in_space(tspace):
                M = selection.build_mask(tspace)
                if tspace == self.space:
                    return self.location.intersects_mask(M)
                else:
                    logger.warning(
                        f"{self.__class__.__name__} cannot be tested for {selection.region.name} in {self.space}, testing in {tspace} instead."
                    )
                    return self.location.warp(tspace).intersects_mask(M)
        else:
            logger.warning(
                f"Cannot test overlap of {self.location} with {selection.region}"
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

    def matches(self, selection):
        """
        Returns true if this feature is linked to the given atlas selection.

        Parameters:
        -----------
        selection : AtlasSelection
        """
        matching_regions = selection.region.find(self.regionspec)
        if len(matching_regions) > 0:
            return True

    def __str__(self):
        return f"{self.__class__.__name__} for {self.regionspec}"


class GlobalFeature(Feature):
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

    def matches(self, selection):
        """
        Returns true if this global feature is related to the given atlas selection.

        Parameters:
        -----------
        selection : AtlasSelection
        """
        if selection.parcellation in self.parcellations:
            return True

    def __str__(self):
        return f"{self.__class__.__name__} for {self.spec}"
