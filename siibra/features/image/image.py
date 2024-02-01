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
"""Base type of features in volume format and related anatomical anchor."""

from zipfile import ZipFile
from .. import feature

from .. import anchor as _anchor

from ...volumes import volume as _volume
from ...volumes.providers import provider

from typing import List


class ImageAnchor(_anchor.AnatomicalAnchor):

    def __init__(self, volume: _volume.Volume, region: str = None):
        _anchor.AnatomicalAnchor.__init__(
            self,
            species=volume.space.species,
            location=None,
            region=region
        )
        self.volume = volume

    @property
    def location(self):
        """
        Loads the bounding box only if required, since it demands image data access.
        """
        if self._location_cached is None:
            self._location_cached = self.volume.get_boundingbox(clip=False)  # use unclipped to preseve exisiting behaviour
        return self._location_cached

    @property
    def space(self):
        return self.volume.space

    def __str__(self):
        return f"Bounding box of image in {self.space.name}"


class Image(feature.Feature, _volume.Volume):

    def __init__(
        self,
        name: str,
        modality: str,
        space_spec: dict,
        providers: List[provider.VolumeProvider],
        region: str = None,
        datasets: List = [],
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=None,  # lazy implementation below!
            anchor=None,  # lazy implementation below!
            datasets=datasets
        )

        _volume.Volume.__init__(
            self,
            space_spec=space_spec,
            providers=providers,
            name=name,
            datasets=datasets,
        )

        self._anchor_cached = ImageAnchor(self, region=region)
        self._description_cached = None
        self._name_cached = name

    def _to_zip(self, fh: ZipFile):
        super()._to_zip(fh)
        # How, what do we download?
        # e.g. for marcel's volume, do we download at full resolution?
        # cannot implement until Volume has an export friendly method
        fh.writestr("volume.txt", "Volume cannot be downloaded yet.")

    @property
    def name(self):
        if self._name_cached is None:
            return feature.Feature.name(self)
        else:
            return f"{self._name_cached} ({self.modality})"

    @property
    def description(self):
        if self._description_cached is None:
            self._description_cached = (
                f"Image feature with modality {self.modality} "
                f"at {self.anchor}"
            )
        return self._description_cached
