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

from . import feature, anchor

from ..volumes import volume
from ..locations import boundingbox

from typing import List


class VolumeOfInterestAnchor(anchor.AnatomicalAnchor):

    def __init__(self, vol: volume.Volume):
        anchor.AnatomicalAnchor.__init__(self, location=None, region=None)
        self.volume = vol

    @property
    def location(self):
        """ loads the bounding box only if required, since it demands image data access. """
        if self._location_cached is None:
            self._location_cached = boundingbox.BoundingBox.from_image(
                self.volume.fetch(), space=self.volume.space
            )
        return self._location_cached

    @property
    def space(self):
        return self.volume.space

    def __str__(self):
        return f"bounding box of volume"


class VolumeOfInterest(feature.Feature, volume.Volume, configuration_folder="features/volumes"):

    def __init__(
        self,
        name: str,
        modality: str,
        space_spec: dict,
        providers: List[volume.VolumeProvider],
        datasets: List = [],
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=None,  # lazy implementation below!
            anchor=None,  # lazy implementation below!
            datasets=datasets
        )
        volume.Volume.__init__(
            self,
            space_spec=space_spec,
            providers=providers,
            name=name,
        )
        self._anchor_cached = VolumeOfInterestAnchor(self)

    @property
    def description(self):
        if self._description_cached is None:
            self._description_cached = (
                f"Volume of interest with modality {self.modality} "
                f"at {self.anchor}"
            )

