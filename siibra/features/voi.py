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

from .. import QUIET
from ..commons import queue_background_task
from ..registry import Preconfigure
from ..volumes.volume import VolumeSrc, VolumeModel, ColorVolumeNotSupported
from ..core.space import BoundingBoxModel, BoundingBox
from ..core.datasets import EbrainsDataset, DatasetJsonModel
from ..retrieval.repositories import GitlabConnector
from ..core.serializable_concept import JSONSerializable

import numpy as np
from typing import List
import hashlib
from pydantic import Field
from os import path


@Preconfigure("features/volumes")
class VolumeOfInterest(SpatialFeature, JSONSerializable):

    def __init__(self, name,  *, volumes=[]):
        if not hasattr(self, 'name'):
            self.name = name
        self.volumes = volumes
        self.background_init_complete = False
        queue_background_task(self._calculate_bounding_box, self)
    
    """
    Since the bounding box is calculated dynamically, it is possible that voi.location is accessed before the property
    is finished initialising. This gates attribute access of "location" until self.background_init_complete is set to True.
    SpatialFeature.__getattribute__ is used to avoid recursion max call stack, suggested by python doc: 
    https://docs.python.org/3/reference/datamodel.html#object.__getattribute__
    """
    def __getattribute__(self, __name: str):
        import time
        if __name == "location":
            while not self.background_init_complete:
                time.sleep(0.5)
        return SpatialFeature.__getattribute__(self, __name)
    
    def _calculate_bounding_box(self, *args, **kwargs):
        assert len(self.volumes) > 0, f"expecting to have at least 1 volumes"
        minpoints = []
        maxpoints = []
        for vsrc in self.volumes:
            try:
                with QUIET:
                    img = vsrc.fetch()
                    D = np.asanyarray(img.dataobj).squeeze()
                    nonzero = np.array(np.where(D > 0))
                    A = img.affine
                minpoints.append(np.dot(A, np.r_[nonzero.min(1)[:3], 1])[:3])
                maxpoints.append(np.dot(A, np.r_[nonzero.max(1)[:3], 1])[:3])

            except ColorVolumeNotSupported:
                # If multi channel volume exists rather than short circuit, try to use other volumes to determine the ROI
                # See PLI hippocampus data feature
                ...
        
        minpoint = np.array(minpoints).min(0)
        maxpoint = np.array(maxpoints).max(0)
        
        space = self.volumes[0].space
        location=BoundingBox(minpoint, maxpoint, space)
        SpatialFeature.__init__(self, location)
        self.background_init_complete = True

    @classmethod
    def _from_json(cls, definition):
        spectype = "siibra/feature/volume/v1.0.0"
        if not definition.get("@type") == spectype:
            raise TypeError(
                f"Received specification of type '{definition.get('@type')}', "
                f"but expected '{spectype}'"
            )

        vsrcs = [VolumeSrc._from_json(vsrc_def) for vsrc_def in definition.get("volumeSrc", [])]

        volume_src_spaces = {vsrc.space for vsrc in vsrcs}
        assert len(volume_src_spaces) == 1, f"expect VolumeOfInterest to have one and only one unique space, but got {len(volume_src_spaces)}"

        kgId = definition.get("kgId")
        if kgId is None:
            return VolumeOfInterest(
                name=definition["name"],
                volumes=vsrcs,
            )
        else:
            return EbrainsVolumeOfInterest(
                dataset_id=definition["kgId"],
                name=definition["name"],
                volumes=vsrcs,
            )

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/voi"

    @property
    def model_id(self):
        _id = hashlib.md5(super().model_id.encode("utf-8")).hexdigest()
        return f"{VolumeOfInterest.get_model_type()}/{str(_id)}"

    def to_model(self, **kwargs) -> "VOIDataModel":
        super_model = super().to_model(**kwargs)
        super_model_dict = super_model.dict()
        super_model_dict["@type"] = VolumeOfInterest.get_model_type()
        return VOIDataModel(
            location=self.location.to_model(**kwargs),
            volumes=[vol.to_model(**kwargs) for vol in self.volumes],
            **super_model_dict,
        )

    @classmethod
    def _bootstrap(cls):
        """
        Load default feature specifications for feature modality.
        """
        conn = GitlabConnector("https://jugit.fz-juelich.de", 3009, "develop")
        for _, loader in conn.get_loaders(folder="vois", suffix=".json"):
            basename = f"{hashlib.md5(loader.url.encode('utf8')).hexdigest()}_{path.basename(_)}"
            cls._add_spec(loader.data, basename)


class EbrainsVolumeOfInterest(VolumeOfInterest, EbrainsDataset):
    def __init__(self, dataset_id, name, volumes, **kwargs):
        EbrainsDataset.__init__(self, dataset_id, name, **kwargs)
        VolumeOfInterest.__init__(self, name, volumes=volumes,**kwargs)


class VOIDataModel(DatasetJsonModel):
    type: str = Field(VolumeOfInterest.get_model_type(), const=True, alias="@type")
    volumes: List[VolumeModel]
    location: BoundingBoxModel
