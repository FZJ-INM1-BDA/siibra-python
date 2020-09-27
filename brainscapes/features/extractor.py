from collections import defaultdict
from abc import ABC
from .feature import Feature

class FeatureExtractor(ABC):
    """
    An abstract class for data features extractors of a particular type,
    related to an atlas.
    """

    _FEATURETYPE = Feature

    def __init__(self):
        self.features = []

    def pick_selection(self,atlas):
        """
        Returns the list of features from this extractor that are associated with
        the selected region of the given atlas object.
        """
        selection = []
        for feature in self.features:
            if feature.matches(atlas):
                selection.append(feature)
        return selection

    def __str__(self):
        return "\n".join([str(f) for f in self.features])

    def register(self,feature):
        assert(isinstance(feature,self._FEATURETYPE))
        self.features.append(feature)


class FeatureExtractorRegistry:

    def __init__(self):
        self._extractors = defaultdict(list)
        self.modalities = {}
        for cls in FeatureExtractor.__subclasses__():
            modality = str(cls._FEATURETYPE).split("'")[1].split('.')[-1]
            self._extractors[modality].append(cls)
            self.modalities[modality] = cls._FEATURETYPE

    def __dir__(self):
        return list(self._extractors.keys())

    def __contains__(self,index):
        return index in self.__dir__()

    def __getattr__(self,name):
        if name in self._extractors.keys():
            return self._extractors[name]
        else:
            raise AttributeError("No such attribute: {}".format(name))
    
    def __getitem__(self,index):
        if index in self._extractors.keys():
            return self._extractors[index]

    def __str__(self):
        return str(self._extractors)
