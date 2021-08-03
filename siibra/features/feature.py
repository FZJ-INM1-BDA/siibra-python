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

from .. import parcellations,logger
from ..region import Region
from ..space import SpaceVOI,SpaceWarper
from typing import Tuple

class Feature:
    """ 
    Abstract base class for all data features.
    """

    def __init__(self,dataset_id):
        self.dataset_id = dataset_id

    def matches(self,atlas):
        """
        Returns True if this feature should be considered part of the current
        selection of the atlas object, otherwise else.
        """
        raise RuntimeError(f"matches(atlas) needs to be implemented by derived classes of {self.__class__.__name__}")

    def __str__(self):
        return f"{self.__class__.__name} feature (id:{self.dataset_id})"

class SpatialFeature(Feature):
    """
    Base class for coordinate-anchored data features.
    """

    def __init__(self,space,dataset_id,location=None):
        """
        Initialize a new spatial feature.
        
        Parameters
        ----------
        space : Space
            The space in which the locations are defined
        dataset_id : str
            Any identifier for the underlying dataset
        location : 3D tuple, or list of 3D tuples
            The 3D physical coordinates in the given space
            Note that the default "None" is  meant to indicate that the feature is not yet full initialized. 
            This is used for multi-point features, where the locations are added manually later on.
        """
        Feature.__init__(self,dataset_id)
        self.space = space
        self.location = location

    @property
    def is_volume_of_interest(self):
        return isinstance(self.location,SpaceVOI)

    @property
    def is_multi_point(self):
        """
        Determines wether this feature is localized by a list of points.
        """
        if self.is_volume_of_interest:
            return False
        elif self.location is None:
            return False
        else:
            return hasattr(self.location[0],"__iter__")

    def matches(self,atlas):
        """
        Returns true if the location information of this feature overlaps with the selected
        region of the atlas, according to the mask in the reference space.
        """
        if self.location is None:
            return False

        elif self.is_multi_point:
            # location is an iterable over locations
            return any([atlas.coordinate_selected(self.space,l) for l in self.location])

        elif self.is_volume_of_interest:
            # If the requested space does not define the selected region, 
            # we try to test in another space.
            for tspace in [self.space]+atlas.spaces:
                if atlas.selected_region.defined_in_space(tspace):
                    M = atlas.build_mask(tspace)
                    if tspace==self.space:
                        return self.location.overlaps(M)
                    else:
                        logger.warn(f"Volume of interest cannot be tested for {atlas.selected_region.name} in {self.space}, testing in {tspace} instead.")
                        minpt = SpaceWarper.convert(self.space,tspace,self.location.minpt)
                        maxpt = SpaceWarper.convert(self.space,tspace,self.location.maxpt)
                        return tspace.get_voi(minpt,maxpt).overlaps(M)

            else:
                logger.warn(f"Cannot test overlap of {self.location} with {atlas.selected_region}")
                return False

        else:
            # location is a single location
            return atlas.coordinate_selected(self.space,self.location)

    def __str__(self):
        xyz2str = lambda c:f"({','.join(map(str,c))})"
        if self.is_multi_point: 
            locstr = f"multiple locations:\n{', '.join(xyz2str(l) for l in self.location)}"
        else:
            locstr = xyz2str(self.location)
        return f"{self.__class__.__name__} in {self.space} space at {locstr}"

class RegionalFeature(Feature):
    """
    Base class for region-anchored data features (semantic anchoring to region
    names instead of coordinates).
    TODO store region as an object that has a link to the parcellation
    """

    def __init__(self,regionspec : Tuple[str,Region],dataset_id:str):
        """
        Parameters
        ----------
        regionspec : string or Region
            Specifier for the brain region, will be matched at test time
        dataset_id : str
            Any identifier for the underlying dataset
        """
        assert(any(map(lambda c:isinstance(regionspec,c),[Region,str])))
        Feature.__init__(self,dataset_id)
        self.regionspec = regionspec
 
    def matches(self,atlas):
        """
        Returns true if this feature is linked to the currently selected region
        in the atlas.
        """
        matching_regions = atlas.selected_region.find(self.regionspec)
        if len(matching_regions)>0:
            return True

    def __str__(self):
        return f"{self.__class__.__name__} for {self.regionspec}"

class GlobalFeature(Feature):
    """
    Base class for data features which apply to the atlas as a whole
    instead of a particular location or region. A typical example is a
    connectivity matrix, which applies to all regions in the atlas.
    """

    def __init__(self,parcellationspec,dataset_id):
        """
        Parameters
        ----------
        parcellationspec : str or Parcellation object
            Identifies the underlying parcellation
        dataset_id : str
            Any identifier for the underlying dataset
        """
        Feature.__init__(self,dataset_id)
        self.spec = parcellationspec
        self.parcellations = parcellations.find(parcellationspec)
 
    def matches(self,atlas):
        """
        Returns true if this global feature is related to the given atlas.
        """
        if atlas.selected_parcellation in self.parcellations:
            return True

