from abc import ABC,abstractmethod
from collections import defaultdict

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
 
    def matches_selection(self,atlas):
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
    """

    def __init__(self,region):
        self.region = region
 
    def matches_selection(self,atlas):
        """
        Returns true if this feature is linked to the currently selected region
        in the atlas.
        """
        matching_regions = atlas.selected_region.find(self.region,exact=False)
        for region in matching_regions:
            if atlas.region_selected(region):
                return True

class FeaturePool(ABC):
    """
    A container for spatial features which implements basic spatial queries.
    """

    _FEATURETYPE = Feature

    def __init__(self):
        self._features = []

    def pick_selection(self,atlas):
        """
        Returns the list of features from this pool that are associated with
        the selected region of the given atlas object.
        """
        selection = []
        for feature in self._features:
            if feature.matches_selection(atlas):
                selection.append(feature)
        return selection

    def __str__(self):
        return "\n".join([str(f) for f in self._features])

    def register(self,feature):
        assert(isinstance(feature,self._FEATURETYPE))
        self._features.append(feature)


class FeaturePoolRegistry:

    def __init__(self):
        self._pools = defaultdict(list)
        for cls in FeaturePool.__subclasses__():
            modality = str(cls._FEATURETYPE).split("'")[1].split('.')[-1]
            self._pools[modality].append(cls)

    def __dir__(self):
        return list(self._pools.keys())

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self._pools.keys():
            return self._pools[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))

    def __getitem__(self,index):
        if index in self._pools.keys():
            return self._pools[index]


    def __str__(self):
        return str(self._pools)
