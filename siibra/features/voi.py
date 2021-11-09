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

from typing import List, Optional

from pydantic.fields import Field
from .feature import SpatialFeature
from .query import FeatureQuery

from .. import QUIET
from ..volumes.volume import VolumeSrc
from ..core.space import Space, BoundingBox, VolumeBaseModel
from ..core.datasets import EbrainsDataset
from ..retrieval.repositories import GitlabConnector
from ..core.jsonable import Literal

import numpy as np

_at_type = 'siibra/spatial-feature/v0.0.1'

class VolumeOfInterest(SpatialFeature, EbrainsDataset):
    def __init__(self, dataset_id, location, name):
        SpatialFeature.__init__(self, location)
        EbrainsDataset.__init__(self, dataset_id, name)
        self.volumes = []

    @classmethod
    def _from_json(cls, definition):
        if definition["@type"] == "minds/core/dataset/v1.0.0":
            space = Space.REGISTRY[definition["space id"]]
            vsrcs = []
            minpoints = []
            maxpoints = []
            for vsrc_def in definition["volumeSrc"]:
                vsrc = VolumeSrc._from_json(vsrc_def)
                vsrc.space = space
                with QUIET:
                    D = vsrc.fetch().get_fdata().squeeze()
                    nonzero = np.array(np.where(D > 0))
                    A = vsrc.build_affine()
                minpoints.append(np.dot(A, np.r_[nonzero.min(1)[:3], 1])[:3])
                maxpoints.append(np.dot(A, np.r_[nonzero.max(1)[:3], 1])[:3])
                vsrcs.append(vsrc)
            minpoint = np.array(minpoints).min(0)
            maxpoint = np.array(maxpoints).max(0)
            result = cls(
                dataset_id=definition["kgId"],
                name=definition["name"],
                location=BoundingBox(minpoint, maxpoint, space),
            )
            list(map(result.volumes.append, vsrcs))
            return result
        return definition

    class VOISerializationSchema(EbrainsDataset.SiibraSerializationSchema):
        at_type: Literal[_at_type] = Field(alias='@type')
        volumes: Optional[List[VolumeBaseModel]]

    SiibraSerializationSchema = VOISerializationSchema

    def to_json(self, detail=False, **kwargs):
        previous_json = super().to_json(**kwargs)
        base = {
            **previous_json,
            '@type': _at_type,
        }
        detailed = {
            'volumes': self.volumes
        } if detail else {}
        return {**base, **detailed}


class VolumeOfInterestQuery(FeatureQuery):
    _FEATURETYPE = VolumeOfInterest
    _QUERY = GitlabConnector("https://jugit.fz-juelich.de", 3009, "develop")

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        for _, loader in self._QUERY.get_loaders(folder="vois", suffix=".json"):
            voi = VolumeOfInterest._from_json(loader.data)  # json.loads(data))
            self.register(voi)
