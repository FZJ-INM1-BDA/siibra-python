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

from .volume import VolumeSrc

from ..retrieval import HttpRequest

import numpy as np


class GiftiSurface(VolumeSrc, volume_type="gii"):
    """
    A (set of) surface meshes in Gifti format.
    """
    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
        VolumeSrc.__init__(self, identifier, name, url, space, detail, **kwargs)
        if isinstance(url, str):
            # a single url
            self._loaders = {name: url}
        elif isinstance(url, dict):
            self._loaders = {
                label: HttpRequest(url)
                for label, url in url.items()
            } 
        else:
            raise NotImplementedError(f"Urls for {self.__class__.__name__} are expected to be of type str or dict.")

    def fetch(self, name=None):
        """
        Returns the mesh as a dictionary with two numpy arrays: An Nx3 array of vertex coordinates, 
        and an Mx3 array of face definitions using row indices of the vertex array.

        If name is specified, only submeshes matching this name are included, otherwise all meshes are combined.
        """
        vertices = np.empty((0, 3))
        faces = np.empty((0,3), dtype='int')
        for n, loader in self._loaders.items():
            npoints = vertices.shape[0]
            if (name is not None) & (n!=name):
                continue
            assert len(loader.data.darrays)>1
            vertices = np.append(vertices, loader.data.darrays[0].data, axis = 0)
            faces = np.append(faces, loader.data.darrays[1].data+npoints, axis = 0)

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


class GiftiSurfaceLabeling(VolumeSrc, volume_type="gii-label"):
    """
    A mesh labeling, specified by a gifti file.
    """
    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
        VolumeSrc.__init__(self, identifier, name, url, space, detail, **kwargs)
        self._loader = HttpRequest(self.url)

    def fetch(self):
        """Returns a 1D numpy array of label indices."""
        assert(len(self._loader.data.darrays)==1)
        return self._loader.data.darrays[0].data


class NeuroglancerMesh(VolumeSrc, volume_type="neuroglancer/precompmesh"):
    """
    A surface mesh provided as neuroglancer precomputed mesh.     
    """

    def __init__(self, identifier, name, url, space, detail=None, **kwargs):
        VolumeSrc.__init__(self, identifier, name, url, space, detail, **kwargs)

    def fetch(self):
        raise NotImplementedError(f"Fetching from neuroglancer precomputed mesh is not yet implemented.")
