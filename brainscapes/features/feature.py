from brainscapes.region import Region
import numpy as np

class SpatialFeature:
    """
    Base class for coordinate-anchored data features.
    """

    def __init__(self,space,location):
        self.space = space
        self.location = location


class SpatialFeaturePool:
    """
    A container for spatial features which implements basic spatial queries.
    """

    def __init__(self):
        self.features = []

    def inside_mask(self,space,mask):
        """
        Returns the features in the pool that can be localized inside the given
        mask of the given template space.
        """
        return [ f 
                for f in self.features
                if (np.all(np.array(self.location)<mask.datashape)mask[f.location[0],f.location[1],f.location[2]]>0]

