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
from ..registry import Preconfigure
from ..volumes.volume import ColorVolumeNotSupported
from ..core.space import BoundingBox
from ..core.datasets import EbrainsDataset
from ..retrieval.repositories import GitlabConnector

import numpy as np
import hashlib
from os import path


@Preconfigure("features/volumes")
class VolumeOfInterest(SpatialFeature):

    def __init__(self, location, name):
        SpatialFeature.__init__(self, location)
        if not hasattr(self, 'name'):
            self.name = name
        self.volumes = []

    @classmethod
    def _from_json(cls, definition):
        spectype = "siibra/feature/volume/v1.0.0"
        if not definition.get("@type") == spectype:
            raise TypeError(
                f"Received specification of type '{definition.get('@type')}', "
                f"but expected '{spectype}'"
            )

        space = None
        vsrcs = []
        minpoints = []
        maxpoints = []
        for vsrc_def in definition["volumeSrc"]:
            try:
                vsrc = VolumeSrc._from_json(vsrc_def)
                if space is None:
                    space = vsrc.space
                else:
                    assert vsrc.space == space
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
            kgId = definition.get("kgId")
            if kgId is None:
                result = VolumeOfInterest(
                    name=definition["name"],
                    location=BoundingBox(minpoint, maxpoint, space),
                )
            else:
                result = EbrainsVolumeOfInterest(
                    dataset_id=definition["kgId"],
                    name=definition["name"],
                    location=BoundingBox(minpoint, maxpoint, space),
                )
            list(map(result.volumes.append, vsrcs))
            return result
        return definition

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
    def __init__(self, dataset_id, location, name):
        EbrainsDataset.__init__(self, dataset_id, name)
        VolumeOfInterest.__init__(self, location, name)

