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

from ..retrieval import requests
from ..commons import logger

import numpy as np
from typing import Union


class GiftiMesh(volume.VolumeProvider, srctype="gii-mesh"):
    """
    One or more surface mesh fragments in Gifti format.
    """

    def __init__(self, url: Union[str, dict], volume=None):
        self.volume = volume
        if isinstance(url, str):  # single mesh
            self._loaders = {None: requests.HttpRequest(url)}
        elif isinstance(url, dict):   # named mesh fragments
            self._loaders = {lbl: requests.HttpRequest(u) for lbl, u in url.items()}
        else:
            raise NotImplementedError(f"Urls for {self.__class__.__name__} are expected to be of type str or dict.")
    
    @property
    def fragments(self):
        return [k for k in self._loaders if k is not None]

    def fetch(self, fragment: str = None, **kwargs):
        """
        Returns the mesh as a dictionary with two numpy arrays: An Nx3 array of vertex coordinates,
        and an Mx3 array of face definitions using row indices of the vertex array.

        A fragment name can be specified to choose from multiple fragments. 
        If not specified, multiple fragments will be merged into one mesh.
        """
        for arg in ["resolution_mm", "voi"]:
            if kwargs.get(arg):
                raise NotImplementedError(f"Parameter {arg} ignored by {self.__class__}.")

        verts = []
        faces = []
        num_verts = 0
        fragments_included = []
        for fragment_name, loader in self._loaders.items():
            if fragment and fragment.lower() not in fragment_name.lower():
                continue
            assert len(loader.data.darrays) > 1
            verts.append(loader.data.darrays[0].data)
            faces.append(loader.data.darrays[1].data + num_verts)
            num_verts += verts[-1].shape[0]
            fragments_included.append(fragment_name)

        if len(fragments_included) > 1:
            logger.info(
               f"The mesh fragments [{', '.join(fragments_included)}] were merged. "
               f"You could select one with the 'fragment' parameter in fetch()."
            )

        return {
            "verts": np.vstack(verts),
            "faces": np.vstack(faces)
        }

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

