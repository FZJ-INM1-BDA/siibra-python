from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, DefaultDict, Literal, Union
from itertools import product
import requests
import re

import numpy as np
import nibabel as nib

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB

from .base import Data
from ..exceptions import InvalidAttrCompException
from ..locations import base, point, pointset
from ..locations.ops.intersection import _loc_intersection
from ..cache import fn_call_cache
from ..retrieval_new.image_fetcher.image_fetcher import (
    get_image_fetcher,
    FetchKwargs,
    MESH_FORMATS,
    VOLUME_FORMATS,
)

if TYPE_CHECKING:
    from ..locations import BBox

IMAGE_VARIANT_KEY = "x-siibra/volume-variant"
IMAGE_FRAGMENT_KEY = "x-siibra/volume-fragment"
HEX_COLOR_REGEXP = re.compile(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$")
SUPPORTED_COLORMAPS = {"magma", "jet", "rgb"}
IMAGE_FORMATS = VOLUME_FORMATS + MESH_FORMATS


def is_hex_color(color: str) -> bool:
    return True if HEX_COLOR_REGEXP.search(color) else False


def check_color(color: str) -> bool:
    if color in SUPPORTED_COLORMAPS or is_hex_color(color):
        return True
    return False


def extract_label_mask(nifti: nib.Nifti1Image, label: int):
    # TODO: Consider adding assertion but this should be clear to the user anyway
    return nib.Nifti1Image((nifti.get_fdata() == label).astype("uint8"), nifti.affine)


@dataclass
class Image(Data, base.Location):
    schema: str = "siibra/attr/data/image/v0.1"
    format: str = None  # see `IMAGE_FORMATS`
    url: str = None
    color: str = None
    subimage_options: DefaultDict[Literal["label", "z"], Union[int, str]] = None

    def __post_init__(self):
        if self.color and not check_color(self.color):
            print(
                f"'{self.color}' is not a hex color or as supported colormap ({SUPPORTED_COLORMAPS=})"
            )

    @staticmethod
    @fn_call_cache
    def _GetBBox(image: "Image"):
        from ..locations import BBox

        if image.format == "neuroglancer/precomputed":
            resp = requests.get(f"{image.url}/info")
            resp.raise_for_status()
            info_json = resp.json()

            resp = requests.get(f"{image.url}/transform.json")
            resp.raise_for_status()
            transform_json = resp.json()

            scale, *_ = info_json.get("scales")
            size = scale.get("size")
            resolution = scale.get("resolution")
            dimension = [s * r for s, r in zip(size, resolution)]
            xs, ys, zs = zip([0, 0, 0], dimension)
            corners = list(product(xs, ys, zs))
            hom = np.c_[corners, np.ones(len(corners))]
            new_coord = np.dot(np.array(transform_json), hom.T)[:3, :].T / 1e6

            min = np.min(new_coord, axis=0)
            max = np.max(new_coord, axis=0)
            return BBox(
                minpoint=min.tolist(), maxpoint=max.tolist(), space_id=image.space_id
            )
        raise NotImplementedError

    @property
    def provides_mesh(self):
        return self.format in MESH_FORMATS

    @property
    def provides_volume(self):
        return self.format in VOLUME_FORMATS

    @property
    def boundingbox(self):
        return Image._GetBBox(self)

    def filter_format(self, format: str):
        if format is None:
            return True
        if format == "mesh":
            return self.format in MESH_FORMATS
        if format == "volume":
            return self.format in VOLUME_FORMATS
        return self.format == format

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
        )
        if color_channel is not None:
            assert self.format == "neuroglancer/precomputed"

        fetcher_fn = get_image_fetcher(self.format)
        nii_or_gii = fetcher_fn(self, fetch_kwargs)

        if self.subimage_options and "label" in self.subimage_options:
            nii_or_gii = extract_label_mask(nii_or_gii, self.subimage_options["label"])

        return nii_or_gii

    def plot(self, *args, **kwargs):
        raise NotImplementedError


def from_nifti(nifti: nib.Nifti1Image, space_id: str) -> "Image":
    """Builds an `Image` `Attribute` from a Nifti image."""
    from hashlib import md5
    from ..cache import CACHE

    filename = CACHE.build_filename(hash(md5(nifti.to_bytes())), suffix=".nii")
    nib.save(nifti, filename)
    return Image(
        fromat="nii",
        url=filename,
        space_id=space_id,
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
