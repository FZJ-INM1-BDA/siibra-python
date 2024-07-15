from dataclasses import dataclass, replace
from typing import Union
import numpy as np
import nibabel as nib
from pathlib import Path
from hashlib import md5

from ...commons import SIIBRA_MAX_FETCH_SIZE_GIB

from .base import Volume
from ...exceptions import InvalidAttrCompException
from ...locations import point, pointset, BBox
from ...locations.ops.intersection import _loc_intersection
from ...retrieval_new.volume_fetcher.volume_fetcher import (
    get_volume_fetcher,
    get_bbox_getter,
    FetchKwargs,
    IMAGE_FORMATS,
)


@dataclass
class Image(Volume):
    schema: str = "siibra/attr/data/image/v0.1"

    def __post_init__(self):
        assert self.format in IMAGE_FORMATS, print(f"{self.format=}")

    @property
    def boundingbox(self) -> "BBox":
        bbox_getter = get_bbox_getter(self.format)
        return bbox_getter(self)

    def fetch(
        self,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
            mapping=self.mapping
        )
        if color_channel is not None:
            assert self.format == "neuroglancer/precomputed"

        fetcher_fn = get_volume_fetcher(self.format)
        return fetcher_fn(self, fetch_kwargs)

    def plot(self, *args, **kwargs):
        raise NotImplementedError


def from_nifti(nifti: Union[str, nib.Nifti1Image], space_id: str, **kwargs) -> "Image":
    """Builds an `Image` `Attribute` from a Nifti image or path to a nifti file."""
    from ...cache import CACHE
    filename = None
    if isinstance(nifti, str):
        filename = nifti
        assert Path(filename).is_file(), f"Provided file {nifti=!r} does not exist"
    if isinstance(nifti, (nib.Nifti1Image, nib.Nifti2Image)):
        filename = CACHE.build_filename(md5(nifti.to_bytes()).hexdigest(), suffix=".nii")
        if not Path(filename).exists():
            nib.save(nifti, filename)
    if not filename:
        raise RuntimeError(f"nifti must be either str or NIftiImage, but you provided {type(nifti)}")
    return Image(
        format="nii",
        url=filename,
        space_id=space_id,
        **kwargs
    )


@_loc_intersection.register(point.Pt, Image)
def compare_pt_to_image(pt: point.Pt, image: Image):
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
