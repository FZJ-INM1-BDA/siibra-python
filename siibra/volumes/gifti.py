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

from . import volume

from ..commons import logger
from ..retrieval import requests
from ..locations import boundingbox

import numpy as np
from typing import Union


class GiftiSurface(volume.VolumeProvider, srctype="gii-mesh"):
    """
    A (set of) surface meshes in Gifti format.
    """

    def __init__(self, url: Union[str, dict], volume=None):
        self.volume = volume
        if isinstance(url, str):
            self._loaders = {"name": requests.HttpRequest(url)}
        elif isinstance(url, dict):
            self._loaders = {lbl: requests.HttpRequest(u) for lbl, u in url.items()}
        else:
            raise NotImplementedError(f"Urls for {self.__class__.__name__} are expected to be of type str or dict.")

    def fetch(self, name=None, resolution_mm: float = None, voi: boundingbox.BoundingBox = None, **kwargs):
        """
        Returns the mesh as a dictionary with two numpy arrays: An Nx3 array of vertex coordinates,
        and an Mx3 array of face definitions using row indices of the vertex array.

        If name is specified, only submeshes matching this name are included, otherwise all meshes are combined.
        """
        if resolution_mm is not None:
            raise NotImplementedError(f"Resolution specification for {self.__class__} not yet implemented.")
        if voi is not None:
            raise NotImplementedError(f"Volume of interest extraction for {self.__class__} not yet implemented.")
        vertices = np.empty((0, 3))
        faces = np.empty((0, 3), dtype='int')
        for n, loader in self._loaders.items():
            npoints = vertices.shape[0]
            if (name is not None) & (n != name):
                continue
            assert len(loader.data.darrays) > 1
            vertices = np.append(vertices, loader.data.darrays[0].data, axis=0)
            faces = np.append(faces, loader.data.darrays[1].data + npoints, axis=0)

        return dict(zip(['verts', 'faces', 'name'], [vertices, faces, name]))

    @property
    def variants(self):
        return list(self._loaders.keys())

    def fetch_iter(self):
        """
        Iterator returning all submeshes, each represented as a dictionary
        with elements
        - 'verts': An Nx3 array of vertex coordinates,
        - 'faces': an Mx3 array of face definitions using row indices of the vertex array
        - 'name': Name of the of the mesh variant
        """
        return (self.fetch(v) for v in self.variants)

    def find_layer_thickness(self, layer0: int = 1, layer1: int = None):
        raise NotImplementedError("Calculation of layer thickness from Gifti meshes is not yet implemented.")


class GiftiSurfaceLabeling(volume.VolumeProvider, srctype="gii-label"):
    """
    A mesh labeling, specified by a gifti file.
    """

    def __init__(self, url: str):
        self._loader = requests.HttpRequest(url)

    def fetch(self, **kwargs):
        """Returns a 1D numpy array of label indices."""
        assert len(self._loader.data.darrays) == 1
        return {"labels": self._loader.data.darrays[0].data}

