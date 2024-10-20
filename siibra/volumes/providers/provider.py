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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Dict, List
from nibabel import Nifti1Image
import json
if TYPE_CHECKING:
    from ...locations.boundingbox import BoundingBox

# TODO add mesh primitive. Check nibabel implementation? Use trimesh? Do we want to add yet another dependency?
VolumeData = Union[Nifti1Image, Dict]


class VolumeProvider(ABC):

    _SUBCLASSES = []

    def __init_subclass__(cls, srctype: str) -> None:
        cls.srctype = srctype
        VolumeProvider._SUBCLASSES.append(cls)
        return super().__init_subclass__()

    @abstractmethod
    def get_boundingbox(self, clip=True, background=0.0) -> "BoundingBox":
        raise NotImplementedError

    @property
    def fragments(self) -> List[str]:
        return []

    @abstractmethod
    def fetch(self, *args, **kwargs) -> VolumeData:
        raise NotImplementedError

    @property
    @abstractmethod
    def _url(self) -> Union[str, Dict[str, str]]:
        """
        This is needed to provide urls to applications that can utilise such resources directly.
        e.g. siibra-api
        """
        return {}


class SubvolumeProvider(VolumeProvider, srctype="subvolume"):
    """
    This provider wraps around an existing volume provider,
    but is preconfigured to always fetch a fixed subvolume.
    The primary use is to provide a fixed z coordinate
    of a 4D volume provider as a 3D volume under the
    interface of a normal volume provider.
    """

    _USE_CACHING = False
    _FETCHED_VOLUMES = {}

    class UseCaching:
        def __enter__(self):
            SubvolumeProvider._USE_CACHING = True

        def __exit__(self, et, ev, tb):
            SubvolumeProvider._USE_CACHING = False
            SubvolumeProvider._FETCHED_VOLUMES = {}

    def __init__(self, parent_provider: VolumeProvider, z: int):
        VolumeProvider.__init__(self)
        self.provider = parent_provider
        self.srctype = parent_provider.srctype
        self.z = z

    def get_boundingbox(self, clip=True, background=0.0, **fetch_kwargs) -> "BoundingBox":
        return self.provider.get_boundingbox(clip=clip, background=background, **fetch_kwargs)

    def fetch(self, **kwargs):
        # activate caching at the caller using "with SubvolumeProvider.UseCaching():""
        if self.__class__._USE_CACHING:
            data_key = json.dumps(self.provider._url, sort_keys=True) \
                + json.dumps(kwargs, sort_keys=True)
            if data_key not in self.__class__._FETCHED_VOLUMES:
                vol = self.provider.fetch(**kwargs)
                self.__class__._FETCHED_VOLUMES[data_key] = vol
            vol = self.__class__._FETCHED_VOLUMES[data_key]
        else:
            vol = self.provider.fetch(**kwargs)
        return vol.slicer[:, :, :, self.z]

    def __getattr__(self, attr):
        return self.provider.__getattribute__(attr)

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return super()._url
