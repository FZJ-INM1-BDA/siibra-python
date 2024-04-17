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
"""Handles reading and preparing gii files."""

from . import provider as _provider

from ...retrieval import requests
from ...commons import logger, merge_meshes
from ...locations import boundingbox as _boundingbox

import numpy as np
from typing import Union, Dict


class GiftiMesh(_provider.VolumeProvider, srctype="gii-mesh"):
    """
    One or more surface mesh fragments in Gifti format.
    """

    def __init__(self, url: Union[str, Dict[str, str]], volume=None):
        self._init_url = url
        self.volume = volume
        # TODO duplicated code to NgMesh
        if isinstance(url, str):  # single mesh
            self._loaders = {None: requests.HttpRequest(url)}
        elif isinstance(url, dict):   # named mesh fragments
            self._loaders = {lbl: requests.HttpRequest(u) for lbl, u in url.items()}
        else:
            raise NotImplementedError(f"Urls for {self.__class__.__name__} are expected to be of type str or dict.")

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._init_url

    def get_boundingbox(self, clip=False, background=0.0, **fetch_kwargs) -> '_boundingbox.BoundingBox':
        """
        Bounding box calculation is not yet implemented for meshes.
        """
        raise NotImplementedError(
            f"Bounding box access to {self.__class__.__name__} objects not yet implemented."
        )

    @property
    def fragments(self):
        return [k for k in self._loaders if k is not None]

    def fetch(self, fragment: str = None, **kwargs):
        """
        Returns the mesh as a dictionary with two numpy arrays.

        Parameters
        ----------
        fragment: str, default: None
            A fragment name can be specified to choose from multiple fragments.

            Note
            ----
            If not specified, multiple fragments will be merged into one mesh.
            In such a case, the verts and faces arrays of different fragments
            are appended to one another.

        Returns
        -------
        dict
            - 'verts': An Nx3 array of vertex coordinates,
            - 'faces': an Mx3 array of face definitions using row indices of the vertex array
        """
        for arg in ["resolution_mm", "voi"]:
            if kwargs.get(arg):
                raise NotImplementedError(f"Parameter {arg} ignored by {self.__class__}.")

        fragments_included = []
        meshes = []
        for fragment_name, loader in self._loaders.items():
            if fragment and fragment.lower() not in fragment_name.lower():
                continue
            assert len(loader.data.darrays) > 1
            meshes.append({
                "verts": loader.data.darrays[0].data,
                "faces": loader.data.darrays[1].data
            })
            fragments_included.append(fragment_name)

        if len(fragments_included) > 1:
            logger.info(
                f"The mesh fragments [{', '.join(fragments_included)}] were merged by "
                "appending vertex information of fragments. "
                f"You could select one with the 'fragment' parameter in fetch()."
            )

        return merge_meshes(meshes)

    @property
    def variants(self):
        return list(self._loaders.keys())

    def fetch_iter(self):
        """
        Iterator returning all submeshes

        Returns
        -------
        dict
            - 'verts': An Nx3 array of vertex coordinates,
            - 'faces': an Mx3 array of face definitions using row indices of the vertex array
            - 'name': Name of the of the mesh variant
        """
        return (self.fetch(v) for v in self.variants)


class GiftiSurfaceLabeling(_provider.VolumeProvider, srctype="gii-label"):
    """
    A mesh labeling, specified by a gifti file.
    """

    def __init__(self, url: Union[str, dict]):
        self._init_url = url
        if isinstance(url, str):  # single mesh labelling
            self._loaders = {None: requests.HttpRequest(url)}
        elif isinstance(url, dict):   # labelling for multiple mesh fragments
            self._loaders = {lbl: requests.HttpRequest(u) for lbl, u in url.items()}
        else:
            raise NotImplementedError(f"Urls for {self.__class__.__name__} are expected to be of type str or dict.")

    def fetch(self, fragment: str = None, label: int = None, **kwargs):
        """Returns a 1D numpy array of label indices."""
        labels = []
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
            assert len(self._loaders[frag].data.darrays) == 1
            if label is not None:
                labels.append((self._loaders[frag].data.darrays[0].data == label).astype('uint8'))
            else:
                labels.append(self._loaders[frag].data.darrays[0].data)

        return {"labels": np.hstack(labels)}

    def get_boundingbox(self, clip=False, background=0.0) -> '_boundingbox.BoundingBox':
        raise NotImplementedError(
            f"Bounding box access to {self.__class__.__name__} objects not yet implemented."
        )

    @property
    def _url(self) -> Union[str, Dict[str, str]]:
        return self._init_url
