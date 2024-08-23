# Copyright 2018-2024
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

from typing import TYPE_CHECKING, Dict, Tuple
import requests

import numpy as np
from nibabel import GiftiImage
from neuroglancer_scripts.mesh import read_precomputed_mesh, affine_transform_mesh

from ...volume_fetcher.volume_fetcher import register_volume_fetcher
from ....cache import fn_call_cache
from ....commons.maps import arrs_to_gii

if TYPE_CHECKING:
    from ....attributes.dataitems import Mesh


@fn_call_cache
def get_mesh_info(url: str) -> Dict:
    return requests.get(f"{url}/info").json()


@fn_call_cache
def get_transform_nm(url: str) -> Dict:
    return requests.get(f"{url}/transform.json").json()


@fn_call_cache
def fetch_mesh_voxels(url: str) -> Tuple[np.ndarray, np.ndarray]:
    req = requests.get(url)
    (vertices_vox, triangles_vox) = read_precomputed_mesh(req.content)
    return vertices_vox, triangles_vox


def ngvoxelmesh_to_gii(
    vertices_vox: np.ndarray, triangles_vox: np.ndarray, transform_nm
):
    vertices, triangles = affine_transform_mesh(
        vertices_vox, triangles_vox, transform_nm
    )
    vertices /= 1e6
    return arrs_to_gii({"verts": vertices, "faces": triangles})


def get_meshindex_info(self, base_url: str, meshindex: int) -> Dict[str, Tuple[str,]]:
    mesh_key = get_mesh_info(base_url).get("mesh")
    meshurl = f"{base_url}/{mesh_key}/{str(meshindex)}:0"
    transform_nm = get_transform_nm(base_url)

    req = requests.get(url=meshurl)
    req.raise_for_status()
    meshdetails = req.json()
    fragment_names = meshdetails.get("fragments")

    if len(fragment_names) == 0:
        raise RuntimeError(f"No fragments found at {meshurl}")
    elif len(self._meshes) > 1:
        # multiple meshes were configured, so we expect only one fragment under each mesh url
        if len(fragment_names) > 1:
            raise RuntimeError(
                f"{self.__class__.__name__} was configured with multiple mesh fragments "
                f"({', '.join(self._meshes.keys())}), but unexpectedly even more fragmentations "
                f"were found at {spec['url']}."
            )
        return (f"{spec['url']}/{mesh_key}/{fragment_names[0]}", transform_nm)
    else:
        # only one mesh was configured, so we might still
        # see muliple fragments under the mesh url
        for fragment_name in fragment_names:
            result[fragment_name] = (
                f"{spec['url']}/{mesh_key}/{fragment_name}",
                transform_nm,
            )


@register_volume_fetcher("neuroglancer/precompmesh", "mesh")
def fetch_neuroglancer_mesh(mesh: "Mesh") -> "GiftiImage":
    vertices_vox, triangles_vox = fetch_mesh_voxels(mesh.url)
    transform_nm = get_transform_nm(mesh.url)
    return ngvoxelmesh_to_gii(vertices_vox, triangles_vox, transform_nm)
