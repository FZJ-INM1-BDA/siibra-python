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

from ..retrieval import HttpRequest

# ahmet
from .. import logger
from ..retrieval.requests import DECODERS
from neuroglancer_scripts import mesh as NGmesh
from io import BytesIO
from nilearn import plotting as nilearn_plotting
# ahmet

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
#
    def __init__(self, url, volume=None):
        VolumeProvider.__init__(self)
        self.volume = volume
        self.url = url

    def fetch(self, resolution_mm=None, name=None):
        """
        Returns the mesh as a dictionary with the keys
        - verticies: an Nx3 array of coordinates (in nanometer)
        - trianlges: an MX3 array containing connection data of verticies
        - name: name of the region 
        """
        def fetch_precomputed_NG_mesh(mesh_url):
            r = HttpRequest(mesh_url, func=lambda b: BytesIO(b))
            (vertices_vox, triangles_vox) = NGmesh.read_precomputed_mesh(r.data)
            vertices, triangles = NGmesh.affine_transform_mesh(vertices_vox, triangles_vox, transform_nm)
            vertices /= 1e6
            return vertices, triangles

        if not self.region.mapped_in_space(space=sapce):
            print('The region is not mapped in this space. Please choose another space.') # replace with exception or warning

        labelled_maps = [self.region.parcellation.get_map(s, "labelled") for s in self.region.supported_spaces]
        map = labelled_maps[0] # how many maps should we expect? shall we give a choice?

        volume_url = VolumeProvider._providers[self.srctype].url

        meshinfo = HttpRequest(url=volume_url + "/info", func=DECODERS['.json']).data
        mesh_key = meshinfo.get('mesh')
        meshes =  HttpRequest(
                url = f"{volume_url}/{mesh_key}/{str(map.get_index(self.region).label)}:0",
                func = DECODERS['.json']
            ).data
            
        mesh_urls = [f"{volume_url}/{mesh_key}/{f}" for f in meshes.get('fragments')]
        
        # raise error when there is no transform.json?
        transform_nm = np.array(HttpRequest(f"{volume_url}/transform.json").data)

        # probably better to run through the list based on metadata from info .json (to determine the keys)
        vertices_l, triangles_l = fetch_precomputed_NG_mesh(mesh_urls[0])
        vertices_r, triangles_r = fetch_precomputed_NG_mesh(mesh_urls[1])
        
        return dict(zip(['left', 'right', 'name'], [{'vertices': vertices_l, 'triangles': triangles_l}, {'vertices': vertices_r, 'triangles': triangles_r}, name]))

    def view_mesh(self, hemisphere: str = 'left'):
        return nilearn_plotting.view_surf((self[hemisphere]['vertices'], self[hemisphere]['triangles']))

    @property
    def variants(self):
        return list(self._loaders.keys())

    def fetch_iter(self):
        return (self.fetch(v) for v in self.variants)