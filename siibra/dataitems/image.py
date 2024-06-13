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
from ..retrieval_new import image_fetcher

if TYPE_CHECKING:
    from ..locations import BBox

IMAGE_VARIANT_KEY = "x-siibra/volume-variant"
IMAGE_FRAGMENT = "x-siibra/volume-fragment"
HEX_COLOR_REGEXP = re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$')
SUPPORTED_COLORMAPS = {"magma", "jet", "rgb"}
VOLUME_FORMATS = [
    "nii",
    "neuroglancer/precomputed",
]
MESH_FORMATS = [
    "neuroglancer/precompmesh",
    "gii-mesh",
    "gii-label",
    "fsaverage-annot",
]
IMAGE_FORMATS = VOLUME_FORMATS + MESH_FORMATS


def fetch_voi(nifti: nib.Nifti1Image, voi: "BBox"):
    bb_vox = voi.transform(np.linalg.inv(nifti.affine))
    (x0, y0, z0), (x1, y1, z1) = bb_vox.minpoint, bb_vox.maxpoint
    shift = np.identity(4)
    shift[:3, -1] = bb_vox.minpoint
    result = nib.Nifti1Image(
        dataobj=nifti.dataobj[x0:x1, y0:y1, z0:z1],
        affine=np.dot(nifti.affine, shift),
    )
    return result


def fetch_label_mask(nifti: nib.Nifti1Image, label: int):
    # TODO: Consider adding assertion but this should be clear to the user anyway
    result = nib.Nifti1Image(
        (nifti.get_fdata() == label).astype('uint8'),
        nifti.affine
    )
    return result


def resample(
    nifti: nib.Nifti1Image,
    resolution_mm: float = None,
    affine=None
):
    # TODO
    # Instead of resmapling nifti to desired resolution_mm in `fetch` as
    # discussed previously, consider an explicit method.
    pass


def is_hex_color(color: str):
    return True if HEX_COLOR_REGEXP.search(color) else False


def check_color(color: str):
    if color in SUPPORTED_COLORMAPS or is_hex_color(color):
        return True
    return False


@dataclass
class Image(Data, base.Location):
    schema: str = "siibra/attr/data/image/v0.1"
    format: str = None  # see `image.IMAGE_FORMATS`
    url: str = None
    color: str = None
    subimage_options: DefaultDict[
        Literal["label", "z"],
        Union[int, str]
    ] = None

    def __post_init__(self):
        if self.format not in image_fetcher.ImageFetcher.SUBCLASSES:
            return
        fetcher_type = image_fetcher.ImageFetcher.SUBCLASSES[self.format]
        self._fetcher = fetcher_type(url=self.url)
        if self.color and not check_color(self.color):
            print(f"'{self.color}' is not a hex color or as supported colormap ({SUPPORTED_COLORMAPS=})")

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

    def fetch(
        self,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None
    ):
        if color_channel is not None:
            assert self.format == "neuroglancer/precomputed"

        img = self._fetcher.fetch()

        if bbox is not None:
            # TODO
            # neuroglancer/precomputed fetches the bbox from the NeuroglancerScale
            img = fetch_voi(img, bbox)

        if self.subimage_options and "label" in self.subimage_options:
            img = fetch_label_mask(img, self.subimage_options["label"])

        if resolution_mm is not None:
            if self.format == "neuroglancer/precomputed":
                pass
            else:
                print(
                    f"Warning: Multi-resolution for '{self.format}' is not supported."
                    "siibra will resample using nilearn to desired resolution"
                )
                img = resample(img, resolution_mm)

        return img

    def plot(self, *args, **kwargs):
        raise NotImplementedError


def from_nifti(nifti: nib.Nifti1Image, space_id: str) -> "Image":
    """Builds an `Image` `Attribute` from a Nifti image."""
    return Image(
        fromat="nii",
        url=None,
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


def intersect_ptcld_image(ptcloud: pointset.PointCloud, image: Image) -> pointset.PointCloud:
    if image.space_id != ptcloud.space_id:
        raise InvalidAttrCompException(
            "ptcloud and image are in different space. Cannot compare the two."
        )

    value_outside = 0

    img = image.data
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
