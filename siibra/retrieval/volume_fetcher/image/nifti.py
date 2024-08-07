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

from typing import TYPE_CHECKING, Tuple, List, Union
import gzip

from nibabel import Nifti1Image
import numpy as np

from ..volume_fetcher import register_volume_fetcher, FetchKwargs

if TYPE_CHECKING:
    from ....attributes.dataitems import Image
    from ....locations import BBox


def create_mask(
    nii: Nifti1Image, background_value: Union[int, float] = 0, lower_threshold: float = None
):
    arr = np.asanyarray(nii.dataobj)
    if lower_threshold is not None:
        return Nifti1Image((arr > lower_threshold).astype("uint8"), nii.affine)
    else:
        return Nifti1Image((arr != background_value).astype("uint8"), nii.affine)


def extract_labels(nii: Nifti1Image, labels: List[int]):
    arr = np.asanyarray(nii.dataobj)
    new_arr = sum([label * (arr == label) for label in labels])
    return Nifti1Image(new_arr, nii.affine)


def extract_float_range(nii: Nifti1Image, range: Tuple[float, float]):
    if not range:
        return nii

    arr = np.asanyarray(nii.dataobj)
    return Nifti1Image(
        (arr[np.where(range[0] < arr < range[1])]).astype(nii.get_data_dtype()),
        nii.affine,
    )


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
    # TODO: Resample nifti to desired resolution_mm. Potentially to use in `fetch`
    raise NotImplementedError


@register_volume_fetcher("nii", "image")
def fetch_nifti(image: "Image", fetchkwargs: FetchKwargs) -> "Nifti1Image":
    if fetchkwargs["color_channel"] is not None:
        raise NotImplementedError

    _bytes = image.get_data()
    try:
        nii = Nifti1Image.from_bytes(gzip.decompress(_bytes))
    except gzip.BadGzipFile:
        nii = Nifti1Image.from_bytes(_bytes)

    if fetchkwargs["bbox"] is not None:
        nii = extract_voi(nii, fetchkwargs["bbox"])

    if fetchkwargs["resolution_mm"] is not None:
        nii = resample(nii, fetchkwargs["resolution_mm"])

    if fetchkwargs["mapping"] is None:
        return nii

    mapping = fetchkwargs["mapping"]
    if mapping is not None:
        details = next(iter(mapping.values()))
        if len(mapping) == 1:
            if "subspace" in details:
                s_ = tuple(
                    slice(None) if isinstance(s, str) else s for s in details["subspace"]
                )
                nii = nii.slicer[s_]
            if "range" in details:
                nii = extract_float_range(nii, details["range"])
        if "label" in details:
            nii = extract_labels(nii, [m["label"] for m in mapping.values()])

    return nii
