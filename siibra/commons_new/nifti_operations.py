from typing import List, TYPE_CHECKING

import numpy as np
from nilearn.image import resample_to_img
from tqdm import tqdm

if TYPE_CHECKING:
    from nibabel import Nifti1Image


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


def resample_to_template_and_merge(
    niftis: List["Nifti1Image"],
    template_img: "Nifti1Image",
    labels: List[int] = []
) -> "Nifti1Image":
    assert len(niftis) > 1, "Need to supply at least two volumes to merge."
    if labels:
        assert len(niftis) == len(labels), "Need to supply as many labels as niftis."

    space = niftis[0].space
    assert all(
        v.space == space for v in niftis
    ), "Cannot merge niftis from different spaces."

    merged_array = np.zeros(template_img.shape, dtype="uint8")

    for i, img in tqdm(
        enumerate(niftis),
        unit=" volume",
        desc=f"Resampling niftis to {space.name} and merging",
        total=len(niftis),
        disable=len(niftis) < 3,
    ):
        resampled_arr = np.asanyarray(resample_img_to_img(img, template_img).dataobj)
        nonzero_voxels = resampled_arr > 0
        if labels:
            merged_array[nonzero_voxels] = labels[i]
        else:
            merged_array[nonzero_voxels] = resampled_arr[nonzero_voxels]

    return Nifti1Image(
        dataobj=merged_array,
        affine=template_img.affine,
        space=space,
        name=f"Resampled and merged niftis: {','.join([v.name for v in niftis])}",
    )
