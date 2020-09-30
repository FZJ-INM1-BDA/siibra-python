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
