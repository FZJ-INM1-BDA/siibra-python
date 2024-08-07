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

from typing import List, Union, Dict, Generator, Tuple
import numpy as np
from nilearn.image import resample_to_img, resample_img
from tqdm import tqdm
from nibabel.nifti1 import Nifti1Image
from nibabel.gifti import GiftiImage, GiftiDataArray
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from ..retrieval.volume_fetcher.image.nifti import create_mask as create_mask_from_nifti


def resample_img_to_img(
    source_img: "Nifti1Image", target_img: "Nifti1Image", interpolation: str = ""
) -> "Nifti1Image":
    """
    Resamples to source image to match the target image according to target's
    affine. (A wrapper of `nilearn.image.resample_to_img`.)

    Parameters
    ----------
    source_img : Nifti1Image
    target_img : Nifti1Image
    interpolation : str, Default: "nearest" if the source image is a mask otherwise "linear".
        Can be 'continuous', 'linear', or 'nearest'. Indicates the resample method.

    Returns
    -------
    Nifti1Image
    """
    interpolation = (
        "nearest" if np.array_equal(np.unique(source_img.dataobj), [0, 1]) else "linear"
    )
    resampled_img = resample_to_img(
        source_img=source_img, target_img=target_img, interpolation=interpolation
    )
    return resampled_img


def affine_scaling(affine):
    """Estimate approximate isotropic scaling factor of an affine matrix."""
    orig = np.dot(affine, [0, 0, 0, 1])
    unit_lengths = []
    for vec in np.identity(3):
        vec_phys = np.dot(affine, np.r_[vec, 1])
        unit_lengths.append(np.linalg.norm(orig - vec_phys))
    return np.prod(unit_lengths)


def merge_volumes(
    volumes: List[Union["Nifti1Image", "GiftiImage"]],
    template_vol: Union["Nifti1Image", "GiftiImage"] = None,
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
    giftis: List["GiftiImage"],
    template_vol: "GiftiImage" = None,
    labels: List[int] = [],
) -> "GiftiImage":
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


def _resample_and_merge_niftis(
    niftis: List["Nifti1Image"],
    template_img: "Nifti1Image" = None,
    labels: List[int] = [],
) -> "Nifti1Image":
    # TODO: get header for affine and shape instead of the whole template
    assert len(niftis) > 1, "Need to supply at least two volumes to merge."
    if labels:
        assert len(niftis) == len(labels), "Need to supply as many labels as niftis."

    if template_img is None:
        shapes = set(nii.shape for nii in niftis)
        assert len(shapes) == 1
        shape = next(iter(shapes))
        merged_array = np.zeros(shape, dtype="int16")
        affine = niftis[0].affine
    else:
        merged_array = np.zeros(template_img.shape, dtype="uint8")
        affine = template_img.affine

    for i, img in tqdm(
        enumerate(niftis),
        unit="nifti",
        desc="Merging (and resmapling if necessary)",
        total=len(niftis),
        disable=len(niftis) < 3,
    ):
        if template_img is not None:
            resampled_arr = np.asanyarray(
                resample_img_to_img(img, template_img).dataobj
            )
        else:
            resampled = resample_img(img, affine, shape)
            resampled_arr = np.asanyarray(resampled.dataobj)
        nonzero_voxels = resampled_arr > 0
        if labels:
            merged_array[nonzero_voxels] = labels[i]
        else:
            merged_array[nonzero_voxels] = resampled_arr[nonzero_voxels]

    return Nifti1Image(dataobj=merged_array, affine=affine)


def spatial_props(
    img: Nifti1Image,
    space_id: str,
    background: float = 0.0,
    maptype: Literal["labelled", "statistical"] = "labelled",
    threshold_statistical: float = None,
):
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
            "centroid": compute_centroid(
                img=Nifti1Image(nonzero, img.affine),
                space_id=space_id,
                background=background,
            ),
            "volume": nonzero.shape[0] * scale,
        }
    return spatialprops


def compute_centroid(img: Nifti1Image, space_id: str, background: float = 0.0):
    from ..attributes.locations import Point

    maparr = np.asanyarray(img.dataobj)
    centroid_vox = np.mean(np.where(maparr != background), axis=1)
    return Point(
        coordinate=np.dot(img.affine, np.r_[centroid_vox, 1])[:3], space_id=space_id
    )


def create_mask(
    volume: Union[Nifti1Image, GiftiImage],
    background_value: Union[int, float] = 0,
    lower_threshold: float = None,
):
    if isinstance(volume, GiftiImage):
        raise NotImplementedError
    if isinstance(volume, Nifti1Image):
        return create_mask_from_nifti(volume, background_value=background_value, lower_threshold=lower_threshold)
