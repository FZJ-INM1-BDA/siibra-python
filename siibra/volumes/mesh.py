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

from .volume import VolumeProvider

from ..commons import logger
from ..retrieval import HttpRequest, SiibraHttpRequestError
from ..retrieval.requests import DECODERS

from neuroglancer_scripts.mesh import read_precomputed_mesh, affine_transform_mesh
from io import BytesIO
import numpy as np


class GiftiSurface(VolumeProvider, srctype="gii-mesh"):
    """
    A (set of) surface meshes in Gifti format.
    """

    def __init__(self, url, volume=None):
        self.volume = volume
        if isinstance(url, str):
            # a single url
            self._loaders = {"name": url}
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


class GiftiSurfaceLabeling():
    """
    A mesh labeling, specified by a gifti file.
    """

    def __init__(self, url):
        self._loader = HttpRequest(self.url)

    def fetch(self):
        """Returns a 1D numpy array of label indices."""
        assert len(self._loader.data.darrays) == 1
        return self._loader.data.darrays[0].data


class NeuroglancerMesh(VolumeProvider, srctype="neuroglancer/precompmesh"):
    """
    A surface mesh provided as neuroglancer precomputed mesh.
    """
    def __init__(self, url, volume=None):
        self.volume = volume
        self.url = url
        self.meshinfo = HttpRequest(url=self.url + "/info", func=DECODERS['.json']).data
        self.mesh_key = self.meshinfo.get('mesh')

    @staticmethod
    def _fetch_fragment(url: str, transform_nm: np.ndarray):
        r = HttpRequest(url, func=lambda b: BytesIO(b))
        (vertices_vox, triangles_vox) = read_precomputed_mesh(r.data)
        vertices, triangles = affine_transform_mesh(vertices_vox, triangles_vox, transform_nm)
        vertices /= 1e6
        return vertices, triangles

    def fetch(self, resolution_mm: float = None, mesh_index: int = None, voi=None):
        """
        Returns the list of fragment meshes found under the given mesh index.
        Each mesh is  a dictionary with the keys:
        - vertices: an Nx3 array of coordinates (in nanometer)
        - faces: an MX3 array containing connection data of vertices
        - name: name of the fragment
        """
        if voi:
            raise RuntimeError("Volume of interests cannot yet be fetched from neuroglancer meshes.")
        try: 
            resp = HttpRequest(
                url=f"{self.url}/{self.mesh_key}/{str(mesh_index)}:0",
                func=DECODERS['.json']
            ).data
            fragments = resp.get('fragments') if resp else None
        except SiibraHttpRequestError:
            logger.error(f"Source {self} does not provide a mesh with index {mesh_index}.")
            return None
        
        transform_nm = np.array(HttpRequest(f"{self.url}/transform.json").data)
        mesh_fragments = [
            self._fetch_fragment(f"{self.url}/{self.mesh_key}/{f}", transform_nm)
            for f in fragments
        ]
        return [
            dict(zip(['verts', 'faces', 'name'], [vertices, faces, fragments[i]]))
            for i, (vertices, faces) in enumerate(mesh_fragments)
        ]

    @property
    def variants(self):
        return list(self._loaders.keys()) # rewrite the code above to have _loaders instead

    def fetch_iter(self):
        return (self.fetch(v) for v in self.variants)