from abc import ABC, abstractmethod
from brainscapes.region import Region

class SpatialFeature:
    """
    Base class for coordinate-anchored data features.
    """

    def __init__(self,space,location):
        self.space = space
        self.location = location

    def in_region(self,region:Region):
        pass


class SpatialFeaturePool:
    """
    A container for spatial features which implements basic spatial queries.
    """

    def __init_(self):
        self.features = []

