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

from dataclasses import dataclass, replace
from typing import Union, Tuple

import numpy as np
import nibabel as nib
from pathlib import Path
from hashlib import md5
from io import BytesIO
import gzip

from ....commons import SIIBRA_MAX_FETCH_SIZE_GIB

from .base import Volume
from ...locations import point, pointset, BoundingBox
from ...locations.ops.intersection import _loc_intersection
from ....retrieval.volume_fetcher.volume_fetcher import (
    get_volume_fetcher,
    get_bbox_getter,
    FetchKwargs,
    IMAGE_FORMATS,
)
from ....commons_new.logger import logger


@dataclass
class Image(Volume):
    schema: str = "siibra/attr/data/image/v0.1"

    def __post_init__(self):
        assert self.format in IMAGE_FORMATS, print(f"{self.format}")

    @property
    def boundingbox(self) -> "BoundingBox":
        bbox_getter = get_bbox_getter(self.format)
        return bbox_getter(self)

    def get_affine(self, **fetch_kwargs: FetchKwargs):
        # TODO: pull from source without fetching the whole image
        return self.fetch(**fetch_kwargs).affine

    def fetch(
        self,
        bbox: "BoundingBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
            mapping=self.mapping,
        )
        if color_channel is not None:
            assert self.format == "neuroglancer/precomputed"

        fetcher_fn = get_volume_fetcher(self.format)
        return fetcher_fn(self, fetch_kwargs)

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def _iter_zippable(self):
        yield from super()._iter_zippable()
        try:
            nii = self.fetch()
            suffix = None
            if isinstance(nii, (nib.Nifti1Image, nib.Nifti2Image)):
                suffix = ".nii.gz"
            if isinstance(nii, nib.GiftiImage):
                suffix = ".gii.gz"
            if not suffix:
                raise RuntimeError("Image.fetch returning a non nifti, non gifti image")

            # TODO not ideal, since loads everything in memory. Ideally we can stream it as IO
            gzipped = gzip.compress(nii.to_bytes())
            yield f"Image (format={self.format})", suffix, BytesIO(gzipped)
        except Exception as e:
            yield f"Image (format={self.format}): to zippable error.", ".error.txt", BytesIO(str(e).encode("utf-8"))


def from_nifti(nifti: Union[str, nib.Nifti1Image], space_id: str, **kwargs) -> "Image":
    """Builds an `Image` `Attribute` from a Nifti image or path to a nifti file."""
    from ....cache import CACHE

    filename = None
    if isinstance(nifti, str):
        filename = nifti
        assert Path(filename).is_file(), f"Provided file {nifti!r} does not exist"
    if isinstance(nifti, (nib.Nifti1Image, nib.Nifti2Image)):
        filename = CACHE.build_filename(
            md5(nifti.to_bytes()).hexdigest(), suffix=".nii"
        )
        if not Path(filename).exists():
            nib.save(nifti, filename)
    if not filename:
        raise RuntimeError(
            f"nifti must be either str or NIftiImage, but you provided {type(nifti)}"
        )
    return Image(format="nii", url=filename, space_id=space_id, **kwargs)


def colorize(
    image: Image, value_mapping: dict, **fetch_kwargs: FetchKwargs
) -> nib.Nifti1Image:
    # TODO: rethink naming
    """
    Create

    Parameters
    ----------
    value_mapping : dict
        Dictionary mapping keys to values

    Return
    ------
    Nifti1Image
    """
    assert image.mapping is not None, ValueError(
        "Provided image must have a mapping defined."
    )

    result = None
    nii = image.fetch(**fetch_kwargs)
    arr = np.asanyarray(nii.dataobj)
    resultarr = np.zeros_like(arr)
    result = nib.Nifti1Image(resultarr, nii.affine)
    for key, value in value_mapping.items():
        assert key in image.mapping, ValueError(
            f"key={key!r} is not in the mapping of the image."
        )
        resultarr[nii == image.mapping[key]["label"]] = value

    return result


@_loc_intersection.register(point.Point, Image)
def compare_pt_to_image(pt: point.Point, image: Image):
    ptcloud = pointset.PointCloud(space_id=pt.space_id, coordinates=[pt.coordinate])
    intersection = compare_ptcloud_to_image(ptcloud=ptcloud, image=image)
    if intersection:
        return pt


@_loc_intersection.register(pointset.PointCloud, Image)
def compare_ptcloud_to_image(ptcloud: pointset.PointCloud, image: Image):
    return intersect_ptcld_image(ptcloud=ptcloud, image=image)


def intersect_ptcld_image(
    ptcloud: pointset.PointCloud, image: Image
) -> pointset.PointCloud:
    if image.space_id != ptcloud.space_id:
        raise InvalidAttrCompException(
            "ptcloud and image are in different space. Cannot compare the two."
        )

    value_outside = 0

    img = image.fetch()
    arr = np.asanyarray(img.dataobj)

    # transform the points to the voxel space of the volume for extracting values
    phys2vox = np.linalg.inv(img.affine)
    voxels = pointset.PointCloud.transform(ptcloud, phys2vox)
    XYZ = np.array(voxels.coordinates).astype("int")

    # temporarily set all outside voxels to (0,0,0) so that the index access doesn't fail
    # TODO in previous version, zero'th voxel is excluded on all sides (i.e. (XYZ > 0) was tested)
    # is there a reason why the zero-th voxel is excluded?
    inside = np.all((XYZ < arr.shape) & (XYZ >= 0), axis=1)
    XYZ[~inside, :] = 0

    # read out the values
    X, Y, Z = XYZ.T
    values = arr[X, Y, Z]

    # fix the outside voxel values, which might have an inconsistent value now
    values[~inside] = value_outside

    inside = list(np.where(values != value_outside)[0])

    return replace(ptcloud, coordinates=[ptcloud.coordinates[i] for i in inside])
