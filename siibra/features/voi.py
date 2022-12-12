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

from .feature import Feature
from . import anchor

from ..volumes.volume import ColorVolumeNotSupported, Volume
from ..core.space import Space
from ..locations import BoundingBox

from typing import List


class VolumeOfInterest(Feature, configuration_folder="features/volumes"):

    def __init__(
        self,
        name: str,
        space_spec: dict,
        measuretype: str,
        volumes: List[Volume],
        datasets: List = [],
    ):
        Feature.__init__(
            self,
            description=None,
            measuretype=measuretype,
            anchor=None,  # lazy implementation below!
            datasets=datasets
        )
        self.volumes = volumes
        self._space_cached = None
        self._space_spec = space_spec

    @property
    def anchor(self):
        if self._anchor_cached is None:
            bbox = None
            for volume in self.volumes:
                next_bbox = BoundingBox.from_image(volume.fetch(), space=self.space)
                if bbox is None:
                    bbox = next_bbox
                else:
                    bbox = bbox.union(next_bbox)
            self._anchor_cached = anchor.AnatomicalAnchor(location=bbox)
        return self._anchor_cached

    @property
    def space(self):
        if self._space_cached is None:
            for key in ["@id", "name"]:
                if key in self._space_spec:
                    self._space_cached = Space.get_instance(self._space_spec[key])
                    break
            else:
                self._space_cached = Space(None, "Unspecified space")
        return self._space_cached
