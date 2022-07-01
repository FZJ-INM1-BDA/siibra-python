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

from .feature import SpatialFeature
from .query import FeatureQuery

from .. import QUIET, logger
from ..volumes.volume import VolumeSrc, VolumeModel, ColorVolumeNotSupported
from ..core.space import BoundingBoxModel, Space, BoundingBox
from ..core.datasets import EbrainsDataset, DatasetJsonModel, OriginDescription
from ..retrieval.repositories import DataproxyConnector, GitlabConnector, DECODERS
from ..core.serializable_concept import JSONSerializable

import numpy as np
from typing import List
import hashlib
from pydantic import Field
from ebrains_drive.exceptions import Unauthorized


class VolumeOfInterest(SpatialFeature, JSONSerializable):
    def __init__(self, location, **kwargs):
        SpatialFeature.__init__(self, location)
        self.volumes = []

    VOI_REGISTRY = {}
    def __init_subclass__(cls, type_id=None):
        if type_id in VolumeOfInterest.VOI_REGISTRY:
            logger.warning(
                f"Type id '{type_id}' already provided by {VolumeOfInterest.VOI_REGISTRY[type_id].__name__}, but {cls.__name__} suggests itself as well"
            )
        if type_id is not None:
            logger.debug(f"Registering specialist {cls.__name__} for type id {type_id}")
            VolumeOfInterest.VOI_REGISTRY[type_id] = cls
        cls.type_id = type_id
        return super().__init_subclass__()
    
    @classmethod
    def _from_json(cls, definition):
        
        def_type = definition.get("@type")
        specific_cls = cls.VOI_REGISTRY.get(def_type)
        if specific_cls:
            space_def = definition.get("space_id") or definition.get("space id")
            space = Space.REGISTRY[space_def]
            vsrcs = []
            minpoints = []
            maxpoints = []
            for vsrc_def in definition["volumeSrc"]:
                try:
                    vsrc = VolumeSrc._from_json(vsrc_def)
                    vsrc.space = space
                    with QUIET:
                        img = vsrc.fetch()
                        D = np.asanyarray(img.dataobj).squeeze()
                        nonzero = np.array(np.where(D > 0))
                        A = img.affine
                    minpoints.append(np.dot(A, np.r_[nonzero.min(1)[:3], 1])[:3])
                    maxpoints.append(np.dot(A, np.r_[nonzero.max(1)[:3], 1])[:3])
                    vsrcs.append(vsrc)
                    
                except ColorVolumeNotSupported:
                    # If multi channel volume exists rather than short circuit, try to use other volumes to determine the ROI
                    # See PLI hippocampus data feature
                    vsrcs.append(vsrc)
                    
            minpoint = np.array(minpoints).min(0)
            maxpoint = np.array(maxpoints).max(0)
            
            extended_definition = {
                **definition,
                'id': definition.get('kgId') or definition.get('@id'),
            }
            
            result = specific_cls(
                location=BoundingBox(minpoint, maxpoint, space),
                **extended_definition
            )
            list(map(result.volumes.append, vsrcs))
            return result
        return definition

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/voi"

    @property
    def model_id(self):
        _id = hashlib.md5(super().model_id.encode("utf-8")).hexdigest()
        return f"{VolumeOfInterest.get_model_type()}/{str(_id)}"

    def to_model(self, **kwargs) -> 'VOIDataModel':
        super_model = super().to_model(**kwargs)
        super_model_dict = super_model.dict()
        super_model_dict["@type"] = VolumeOfInterest.get_model_type()
        return VOIDataModel(
            location=self.location.to_model(**kwargs),
            volumes=[vol.to_model(**kwargs) for vol in self.volumes],
            **super_model_dict,
        )

class SimpleVolumeOfInterest(VolumeOfInterest, OriginDescription, type_id="fzj/tmp/simpleOriginInfo/v0.0.1"):
    def __init__(self, *args, **kwargs):
        VolumeOfInterest.__init__(self, *args, **kwargs)
        OriginDescription.__init__(self, *args, **kwargs)


class EbrainsVolumeOfInterest(VolumeOfInterest, EbrainsDataset, type_id="minds/core/dataset/v1.0.0"):
    def __init__(self, *args, **kwargs):
        VolumeOfInterest.__init__(self, *args, **kwargs)
        EbrainsDataset.__init__(self, *args, **kwargs)


class VOIDataModel(DatasetJsonModel):
    type: str = Field(VolumeOfInterest.get_model_type(), const=True, alias="@type")
    volumes: List[VolumeModel]
    location: BoundingBoxModel


class VolumeOfInterestQuery(FeatureQuery):
    _FEATURETYPE = VolumeOfInterest
    _QUERY = GitlabConnector("https://jugit.fz-juelich.de", 3009, "develop")

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        for _, loader in self._QUERY.get_loaders(folder="vois", suffix=".json"):
            try:
                voi = VolumeOfInterest._from_json(loader.data)  # json.loads(data))
                self.register(voi)
            except Exception as e:
                logger.warn(f"some VOI cannot be loaded: {str(e)}")
        try:
            query = DataproxyConnector()
            for _, loader in query.get_loaders(folder="features/volumes", suffix=".json"):
                try:
                    json_content = DECODERS['.json'](loader.data)
                    
                    json_output = json_content.get("output", {}).get("siibra-python")
                    assert json_output is not None, f"dataproxy connector fetched feature/volume/.json should contain '.output.siibra-python', but does not. {loader.data}"
                    
                    voi = VolumeOfInterest._from_json(json_output)
                    self.register(voi)
                except AssertionError as e:
                    logger.warn(f"Assertion error: {str(e)}")
                except Exception as e:
                    # Misc exception?
                    logger.debug(f"Misc exception: {str(e)}")
                    
        except Unauthorized as e:
            # provided token cannot access
            # ignore and carry on
            logger.debug(f"Could not fetch siibra-configuration: {str(e)}")
        except RuntimeError as e:
            # if token has not been passed, run time error will be raised.
            # ignore and carry on
            pass
