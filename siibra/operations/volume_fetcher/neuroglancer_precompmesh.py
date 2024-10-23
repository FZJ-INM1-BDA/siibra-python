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

from typing import Dict, Tuple, TYPE_CHECKING, List
import requests
import numpy as np
from nibabel import GiftiImage
from neuroglancer_scripts.mesh import read_precomputed_mesh, affine_transform_mesh
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from .base import VolumeFormats
from ...cache import fn_call_cache
from ...commons.maps import arrs_to_gii
from ...operations import DataOp

if TYPE_CHECKING:
    from ...attributes.datarecipes.volume import VolumeRecipe


@fn_call_cache
def get_mesh_info(url: str) -> Tuple[Dict, List[List[float]]]:
    sess = requests.Session()
    return sess.get(f"{url}/info").json(), sess.get(f"{url}/transform.json").json()


def ngvoxelmesh_to_gii(
    vertices_vox: np.ndarray, triangles_vox: np.ndarray, transform_nm
):
    vertices, triangles = affine_transform_mesh(
        vertices_vox, triangles_vox, transform_nm
    )
    vertices /= 1e6
    return arrs_to_gii({"verts": vertices, "faces": triangles})


@VolumeFormats.register_format_read("neuroglancer/precompmesh", "mesh")
def read_freesurfer_annot(conf: Dict, _: List[Dict]):
    base_url = conf.get("url")
    assert base_url, f"Expected url attribute to be in {conf}, but was not"

    label = (conf.get("archive_options") or {}).get("label")
    assert label, f"expected archive_options.label to be in {conf}, but was not"

    from .gifti import MergeGifti

    return [
        ReadNgMeshes.generate_specs(base_url=base_url, label=label),
        MergeGifti.generate_specs(),
    ]


class ReadNgMeshes(DataOp):
    input: None
    output: List[GiftiImage]
    desc = "Given a neuroglancer base_url, a label index, return a list of GiftiMeshes"
    type = "read/neuroglancer_precompmesh"

    def run(self, input, base_url: str, label: int, **kwargs):
        assert label is not None, "index label must be provided for ReadNgMeshes"
        assert base_url is not None, "base_url must be provided for ReadNgMeshes"
        sess = requests.Session()

        info_json, transform_nm = get_mesh_info(base_url)

        mesh_dir = info_json.get("mesh")
        assert mesh_dir, f"{base_url} does not have mesh key defined."

        mesh_info_resp = sess.get(f"{base_url}/{mesh_dir}/{str(label)}:0")
        mesh_info_resp.raise_for_status()
        mesh_info_json = mesh_info_resp.json()
        fragments = mesh_info_json.get("fragments")

        def fetch_mesh_voxels(url: str) -> Tuple[np.ndarray, np.ndarray]:
            req = sess.get(url)
            vertices_vox, triangles_vox = read_precomputed_mesh(BytesIO(req.content))
            return vertices_vox, triangles_vox

        with ThreadPoolExecutor() as ex:
            meshes = ex.map(
                fetch_mesh_voxels,
                [f"{base_url}/{mesh_dir}/{frag}" for frag in fragments],
            )

        result = []
        for vertices_vox, triangles_vox in meshes:
            gii = ngvoxelmesh_to_gii(
                vertices_vox, triangles_vox, np.array(transform_nm)
            )
            result.append(gii)
        return result

    @classmethod
    def generate_specs(cls, *, base_url: str, label: int = None, **kwargs):
        base = super().generate_specs(**kwargs)
        return {**base, "base_url": base_url, "label": label}
