from typing import TYPE_CHECKING
import gzip

from nibabel import Nifti1Image
import numpy as np

from ..volume_fetcher import register_volume_fetcher, FetchKwargs

if TYPE_CHECKING:
    from ....dataitems import Image
    from ....locations import BBox


def extract_voi(nifti: Nifti1Image, voi: "BBox"):
    bb_vox = voi.transform(np.linalg.inv(nifti.affine))
    (x0, y0, z0), (x1, y1, z1) = bb_vox.minpoint, bb_vox.maxpoint
    shift = np.identity(4)
    shift[:3, -1] = bb_vox.minpoint
    result = Nifti1Image(
        dataobj=nifti.dataobj[x0:x1, y0:y1, z0:z1],
        affine=np.dot(nifti.affine, shift),
    )
    return result


def resample(nifti: Nifti1Image, resolution_mm: float = None, affine=None):
    # TODO
    # Instead of resmapling nifti to desired resolution_mm in `fetch` as
    # discussed previously, consider an explicit method.
    raise NotImplementedError


@register_volume_fetcher("nii", "image")
def fetch_nifti(image: "Image", fetchkwargs: FetchKwargs) -> "Nifti1Image":
    _bytes = image.get_data()
    try:
        nii = Nifti1Image.from_bytes(gzip.decompress(_bytes))
    except gzip.BadGzipFile:
        nii = Nifti1Image.from_bytes(_bytes)

    if fetchkwargs["bbox"] is not None:
        # TODO
        # neuroglancer/precomputed fetches the bbox from the NeuroglancerScale
        nii = extract_voi(nii, fetchkwargs["bbox"])

    if fetchkwargs["resolution_mm"] is not None:
        nii = resample(nii, fetchkwargs["resolution_mm"])

    return nii