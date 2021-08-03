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

import numpy as np

from .. import spaces,QUIET
from ..volumesrc import VolumeSrc
from ..space import SpaceVOI
from .feature import SpatialFeature
from .query import FeatureQuery
from ..retrieval import GitlabConnector


class VolumeOfInterest(SpatialFeature):

    def __init__(self,space,dataset_id,location,name):
        SpatialFeature.__init__(
            self,
            space=space,
            dataset_id=dataset_id,
            location=location)
        self.name = name
        self.volumes = []
    
    @staticmethod
    def from_json(definition):
        if definition["@type"]=="minds/core/dataset/v1.0.0":
            space = spaces[definition['space id']]
            vsrcs = []
            minpts = []
            maxpts = []            
            for vsrc_def in definition["volumeSrc"]:
                vsrc = VolumeSrc.from_json(vsrc_def)
                vsrc.space = space
                with QUIET:
                    D = vsrc.fetch().get_fdata().squeeze()
                    nonzero = np.array(np.where(D>0))
                    A = vsrc.build_affine()
                minpts.append(np.dot(A,np.r_[nonzero.min(1)[:3],1])[:3])
                maxpts.append(np.dot(A,np.r_[nonzero.max(1)[:3],1])[:3])
                vsrcs.append(vsrc)
            minpt=np.array(minpts).min(0)
            maxpt=np.array(maxpts).max(0)
            result = VolumeOfInterest(
                space, 
                dataset_id = definition['kgId'],
                name = definition['name'],
                location=SpaceVOI(space,minpt,maxpt))
            list(map(result.volumes.append,vsrcs))
            return result
        return definition
    
class VolumeOfInterestQuery(FeatureQuery):
    _FEATURETYPE = VolumeOfInterest
    _QUERY = GitlabConnector("https://jugit.fz-juelich.de",3009,"develop")

    def __init__(self):
        FeatureQuery.__init__(self)
        for _,loader in self._QUERY.get_loaders(folder="vois",suffix=".json"):
            voi = VolumeOfInterest.from_json(loader.data)#json.loads(data))
            self.register(voi)
