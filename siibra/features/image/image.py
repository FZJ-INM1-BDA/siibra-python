# Copyright 2018-2026
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
"""Base type of features in volume format."""

from functools import partial
from typing import List, TYPE_CHECKING
from zipfile import ZipFile

from .. import feature
from .. import anchor as _anchor
from ...volumes import volume as _volume

if TYPE_CHECKING:
    from ...locations.boundingbox import BoundingBox
    from ...volumes.providers import provider


class Image(
    feature.Feature,
    _volume.Volume,
    configuration_folder="features/images",
    category="generic"
):

    def __init__(
        self,
        name: str,
        modality: str,
        space_spec: dict,
        providers: List["provider.VolumeProvider"],
        region: str = None,
        datasets: List = [],
        bbox: "BoundingBox" = None,
        id: str = None,
        prerelease: bool = False,
        time: List[float] = None
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=None,  # lazy implementation below!
            anchor=None,  # lazy implementation below!
            datasets=datasets,
            id=id,
            prerelease=prerelease,
        )

        vol_kwargs = dict(
            space_spec=space_spec,
            providers=providers,
            name=name,
            datasets=datasets,
            bbox=bbox,
        )
        if time is not None:
            _volume.TimeSeriesVolume.__init__(self, time=time, **vol_kwargs)
        else:
            _volume.Volume.__init__(self, **vol_kwargs)

        # Anchored to the image's space and, if given, its region. The bounding
        # box locates it for location queries and for region queries without
        # a region; computing it may need the image, so only when needed.
        self._anchor_cached = _anchor.AnatomicalAnchor(
            species=self.space.species,
            region=region,
            space=self.space,
            location=partial(self.get_boundingbox, clip=False),
        )
        self._description_cached = None

    def _to_zip(self, fh: ZipFile):
        super()._to_zip(fh)
        # How, what do we download?
        # e.g. for marcel's volume, do we download at full resolution?
        # cannot implement until Volume has an export friendly method
        fh.writestr("volume.txt", "Volume cannot be downloaded yet.")

    @property
    def name(self):
        if self._name is None:
            return feature.Feature.name.fget(self)
        name = f"{self._name} ({self.modality})"
        return f"[PRERELEASE] {name}" if self._prerelease else name

    @property
    def description(self):
        if self._description_cached is None:
            self._description_cached = (
                f"Image feature with modality {self.modality}, {self.anchor}"
            )
        return self._description_cached
