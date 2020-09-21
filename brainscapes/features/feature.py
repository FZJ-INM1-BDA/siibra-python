import numpy as np
from abc import ABC,abstractmethod

class Feature(ABC):
    """ 
    Abstract base class for all data features.
    """

    @abstractmethod
    def matches_selection(self,atlas):
        """
        Returns True if this feature should be considered part of the current
        selection of the atlas object, otherwise else.
        """
        pass

class SpatialFeature(Feature):
    """
    Base class for coordinate-anchored data features.
    """

    def __init__(self,space,location):
        self.space = space
        self.location = location
 
    def matches_selection(self,atlas):
        """
        Returns true if the location of this feature is inside the selected
        region of the atlas, according to the mask in the reference space.
        """
        return atlas.inside_selection(self.space,self.location)


class FeaturePool:
    """
    A container for spatial features which implements basic spatial queries.
    """

    def __init__(self):
        self.features = []

    def pick_selection(self,atlas):
        """
        Returns the list of features from this pool that are associated with
        the selected region of the given atlas object.
        """
        selection = []
        for feature in self.features:
            if feature.matches_selection(atlas):
                selection.append(feature)
        return selection

