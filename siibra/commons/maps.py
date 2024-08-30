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

from typing import List, Union, Dict, TYPE_CHECKING
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nibabel.gifti import GiftiImage, GiftiDataArray

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

from ..operations import DataOp
from ..attributes.locations import Point


def affine_scaling(affine):
    """Estimate approximate isotropic scaling factor of an affine matrix."""
    orig = np.dot(affine, [0, 0, 0, 1])
    unit_lengths = []
    for vec in np.identity(3):
        vec_phys = np.dot(affine, np.r_[vec, 1])
        unit_lengths.append(np.linalg.norm(orig - vec_phys))
    return np.prod(unit_lengths)


def merge_volumes(
    volumes: List[Union[Nifti1Image, GiftiImage]],
    template_vol: Union[Nifti1Image, GiftiImage] = None,
    labels: List[int] = [],
):
    vol_types = {type(vol) for vol in volumes}
    assert len(vol_types) == 1

    if Nifti1Image in vol_types:
        return _resample_and_merge_niftis(volumes, template_vol, labels)

    if GiftiImage in vol_types:
        return _merge_giftis(volumes, template_vol, labels)


def gii_to_arrs(gii: GiftiImage) -> Dict[str, np.ndarray]:
    mesh = dict()
    for arr in gii.darrays:
        if arr.intent == 1008:
            mesh.update({"verts": arr.data})
        if arr.intent == 1009:
            mesh.update({"faces": arr.data})
        if arr.intent == 2005:
            mesh.update({"labels": arr.data})
    return mesh


def arrs_to_gii(mesh: Dict[str, np.ndarray]) -> "GiftiImage":
    darrays = []
    for key, arr in mesh.items():
        if key == "verts":
            darrays.append(GiftiDataArray(arr, intent=1008))
        if key == "faces":
            darrays.append(GiftiDataArray(arr, intent=1009))
        if key == "labels":
            darrays.append(GiftiDataArray(arr, intent=2005)).astype("int32")
    return GiftiImage(darrays=darrays)


def _merge_giilabels(giftis: List["GiftiImage"]) -> "GiftiImage":
    labels = [gii_to_arrs(gii)["labels"] for gii in giftis]
    return GiftiImage(
        darrays=GiftiDataArray(np.hstack(labels).astype("int32"), intent=2005)
    )


def _merge_giftis(
    giftis: List[GiftiImage],
    template_vol: GiftiImage = None,
    labels: List[int] = [],
) -> GiftiImage:
    meshes = [gii_to_arrs(gii) for gii in giftis]
    assert len(meshes) > 0
    if len(meshes) == 1:
        return meshes[0]

    try:
        assert all("verts" in m for m in meshes)
        assert all("faces" in m for m in meshes)
    except AssertionError:
        assert all("labels" in m for m in meshes)
        merged_labelled_gii = _merge_giilabels(giftis)
        if template_vol is None:
            return merged_labelled_gii
        merged_gii = template_vol
        merged_gii.darrays.append(merged_labelled_gii.darrays)
        return merged_gii

    has_labels = all("labels" in m for m in meshes)
    if has_labels:
        assert len(labels) == 0

    nverts = [0] + [m["verts"].shape[0] for m in meshes[:-1]]
    verts = np.concatenate([m["verts"] for m in meshes])
    faces = np.concatenate([m["faces"] + N for m, N in zip(meshes, nverts)])
    if has_labels:
        labels = np.array([_ for m in meshes for _ in m["labels"]])
        return arrs_to_gii({"verts": verts, "faces": faces, "labels": labels})
    elif len(labels) != 0:
        assert len(labels) == len(meshes)
        labels = np.array([labels[i] for i, m in enumerate(meshes) for v in m["verts"]])
        return arrs_to_gii({"verts": verts, "faces": faces, "labels": labels})
    else:
        return arrs_to_gii({"verts": verts, "faces": faces})


class SpatialProp(TypedDict):
    centroid: "Point"
    volume: int


def spatial_props(
    img: Nifti1Image,
    background: float = 0.0,
    maptype: Literal["labelled", "statistical"] = "labelled",
    threshold_statistical: float = None,
) -> Dict[int, SpatialProp]:
    from skimage import measure

    spatialprops = {}

    # determine scaling factor from voxels to cube mm
    scale = affine_scaling(img.affine)

    if threshold_statistical:
        assert (
            maptype == "statistical"
        ), "`threshold_statistical` can only be used for statistical maps."
        arr_ = np.asanyarray(img.dataobj)
        arr = (arr_ > threshold_statistical).astype("uint8")
    else:
        arr = np.asanyarray(img.dataobj)
    # compute properties of labelled volume
    A = np.asarray(arr, dtype=np.int32).squeeze()
    C = measure.label(A)

    # compute spatial properties of each connected component
    for label in range(1, C.max() + 1):
        nonzero = np.c_[np.nonzero(C == label)]
        spatialprops[label] = {
            "centroid": ComputeCentroid.run(
                input=Nifti1Image(nonzero, img.affine),
                background_value=background,
            ),
            "volume": nonzero.shape[0] * scale,
        }
    return spatialprops


# TODO move to dataops/volumesops ?
class ComputeCentroid(DataOp):
    input: Nifti1Image
    output: Point
    desc = "Transforms nifti to nifti"
    type = "nifti/compute/centroid"

    def run(self, input, *, background_value=0, **kwargs):
        assert isinstance(
            input, Nifti1Image
        ), f"Expected input to be of type nifti1image, but was {type(input)}"
        maparr = np.asanyarray(input.dataobj)
        centroid_vox = np.mean(np.where(maparr != background_value), axis=1)
        return Point(coordinate=np.dot(input.affine, np.r_[centroid_vox, 1])[:3])

    @classmethod
    def generate_specs(cls, background_value=0, **kwargs):
        base = super().generate_specs(**kwargs)
        return {**base, "background_value": background_value}


def create_mask(
    volume: Union[Nifti1Image, GiftiImage, None],
    background_value: Union[int, float] = 0,
    lower_threshold: float = None,
):
    if isinstance(volume, GiftiImage):
        raise NotImplementedError
    if isinstance(volume, Nifti1Image):
        return create_mask_from_nifti(
            volume, background_value=background_value, lower_threshold=lower_threshold
        )
    raise RuntimeError(
        f"volume must be of type nifti or gifti, but go {type(volume).__name__}"
    )
