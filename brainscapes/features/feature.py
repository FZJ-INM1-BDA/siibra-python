# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC,abstractmethod
from .. import logger

class Feature(ABC):
    """ 
    Abstract base class for all data features.
    """

    @abstractmethod
    def matches(self,atlas):
        """
        Returns True if this feature should be considered part of the current
        selection of the atlas object, otherwise else.
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Print a reasonable name of this feature.
        """
        return 

class SpatialFeature(Feature):
    """
    Base class for coordinate-anchored data features.
    """

    def __init__(self,space,location):
        self.space = space
        self.location = location
 
    def matches(self,atlas):
        """
        Returns true if the location of this feature is inside the selected
        region of the atlas, according to the mask in the reference space.
        """
        return atlas.coordinate_selected(self.space,self.location)

    def __str__(self):
        return "Features in '{space}' at {loc[0]}/{loc[1]}/{loc[2]}".format(
                space=self.space, loc=self.location)

class RegionalFeature(Feature):
    """
    Base class for region-anchored data features (semantic anchoring to region
    names instead of coordinates).
    TODO store region as an object that has a link to the parcellation
    """

    def __init__(self,region):
        self.region = region
 
    def matches(self,atlas):
        """
        Returns true if this feature is linked to the currently selected region
        in the atlas.
        """
        if not atlas.selected_region:
            logger.warning("No region selected in atlas - cannot filter features.")
            return False
        matching_regions = atlas.selected_region.find(self.region)
        for region in matching_regions:
            if atlas.region_selected(region):
                return True

class GlobalFeature(Feature):
    """
    Base class for data features which apply to the atlas as a whole
    instead of a particular location or region. A typical example is a
    connectivity matrix, which applies to all regions in the atlas.
    """

    def __init__(self,parcellation):
        self.parcellation = parcellation
 
    def matches(self,atlas):
        """
        Returns true if this global feature is related to the given atlas.
        """
        if self.parcellation == atlas.selected_parcellation:
            return True

