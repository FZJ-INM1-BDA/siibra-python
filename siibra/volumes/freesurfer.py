# Copyright 2018-2025
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
"""Handles reading and preparing gii files."""

from typing import Union, Dict, TYPE_CHECKING

import numpy as np

from . import volume
from ..retrieval.requests import HttpRequest, ZipfileRequest

if TYPE_CHECKING:
    from ..locations import boundingbox as _boundingbox


class FreesurferAnnot(volume.VolumeProvider, srctype="freesurfer-annot"):
    def __init__(self, url: Union[str, dict]):
        self._init_url = url
        if isinstance(url, str):  # single mesh labelling
            self._loaders = {None: HttpRequest(url)}
        elif isinstance(url, dict):   # named label fragments
            self._loaders = {lbl: HttpRequest(u) for lbl, u in url.items()}
        else:
            raise NotImplementedError(f"Urls for {self.__class__.__name__} are expected to be of type str.")

    def fetch(self, fragment: str = None, label: int = None, **kwargs):
        """Returns a 1D numpy array of label indices."""
        vertex_labels = []
        if fragment is None:
            matched_frags = list(self._loaders.keys())
        else:
            matched_frags = [frg for frg in self._loaders.keys() if fragment.lower() in frg.lower()]
            if len(matched_frags) != 1:
                raise ValueError(
                    f"Requested fragment '{fragment}' could not be matched uniquely "
                    f"to [{', '.join(self._loaders)}]"
                )
        for frag in matched_frags:
            frag_labels, *_ = self._loaders[frag].data
            if label is not None:  # create the mask
                selected_label = frag_labels == label
                frag_labels[selected_label] = 1
                frag_labels[~selected_label] = 0
            else:
                frag_labels[frag_labels == -1] = 0  # annot files store background as -1 while siibra uses 0
            vertex_labels.append(frag_labels)

        return {"labels": np.hstack(vertex_labels)}

    @property
    def boundingbox(self) -> '_boundingbox.BoundingBox':
        raise NotImplementedError(
            f"Bounding box access to {self.__class__.__name__} objects not yet implemented."
        )

    @property
    def fragments(self):
        return [k for k in self._loaders if k is not None]

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._init_url


class ZippedFreesurferAnnot(volume.VolumeProvider, srctype="zip/freesurfer-annot"):
    def __init__(self, url: Union[str, dict]):
        self._init_url = url
        if isinstance(url, str):  # single mesh labelling
            self._loaders = {None: ZipfileRequest(*url.split(" "))}
        elif isinstance(url, dict):   # named label fragments
            self._loaders = {lbl: ZipfileRequest(*u.split(" ")) for lbl, u in url.items()}
        else:
            raise NotImplementedError(f"Urls for {self.__class__.__name__} are expected to be of type str.")

    def fetch(self, fragment: str = None, label: int = None, **kwargs):
        """Returns a 1D numpy array of label indices."""
        vertex_labels = []
        if fragment is None:
            matched_frags = list(self._loaders.keys())
        else:
            matched_frags = [frg for frg in self._loaders.keys() if fragment.lower() in frg.lower()]
            if len(matched_frags) != 1:
                raise ValueError(
                    f"Requested fragment '{fragment}' could not be matched uniquely "
                    f"to [{', '.join(self._loaders)}]"
                )
        for frag in matched_frags:
            frag_labels, *_ = self._loaders[frag].data
            if label is not None:  # create the mask
                selected_label = frag_labels == label
                frag_labels[selected_label] = 1
                frag_labels[~selected_label] = 0
            else:
                frag_labels[frag_labels == -1] = 0  # annot files store background as -1 while siibra uses 0
            vertex_labels.append(frag_labels)

        return {"labels": np.hstack(vertex_labels)}

    @property
    def boundingbox(self) -> '_boundingbox.BoundingBox':
        raise NotImplementedError(
            f"Bounding box access to {self.__class__.__name__} objects not yet implemented."
        )

    @property
    def fragments(self):
        return [k for k in self._loaders if k is not None]

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._init_url
