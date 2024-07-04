from typing import List, Literal
import numpy as np
from nilearn.image import resample_to_img, resample_img
from tqdm import tqdm
from nibabel import Nifti1Image

from ..locations import Pt


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


def resample_and_merge(
    niftis: List["Nifti1Image"],
    template_img: "Nifti1Image" = None,
    labels: List[int] = [],
) -> "Nifti1Image":
    # TODO: must handle meshes
    # TODO: get header for affine and shape instead of the whole template
    assert len(niftis) > 1, "Need to supply at least two volumes to merge."
    if labels:
        assert len(niftis) == len(labels), "Need to supply as many labels as niftis."

    if template_img is None:
        shapes = set(nii.shape for nii in niftis)
        assert len(shapes) == 1
        shape = next(iter(shapes))
        merged_array = np.zeros(shape, dtype="uint8")
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
            "volume": nonzero.shape[0] * scale
        }
    return spatialprops


def compute_centroid(img: Nifti1Image, space_id: str, background: float = 0.0):
    maparr = np.asanyarray(img.dataobj)
    centroid_vox = np.mean(np.where(maparr != background), axis=1)
    return Pt(
        coordinate=np.dot(img.affine, np.r_[centroid_vox, 1])[:3], space_id=space_id
    )
