from dataclasses import dataclass, replace
from typing import Literal, Union
import requests
import nibabel as nib
from pathlib import Path
from itertools import product
import numpy as np

from .base import Data
from ..exceptions import InvalidAttrCompException
from ..locations import base, Pt, PointCloud
from ..locations.ops.intersection import _loc_intersection
from ..cache import CACHE, fn_call_cache

IMAGE_VARIANT_KEY = "x-siibra/volume-variant"
IMAGE_FRAGMENT = "x-siibra/volume-fragment"

MESH_FORMATS = (
    "neuroglancer/precompmesh",
    "gii-mesh",
)

VOLUME_FORMATS = {
    "nii",
    "neuroglancer/precomputed",
}


@dataclass
class Image(Data, base.Location):

    schema: str = "siibra/attr/data/image/v0.1"
    format: Literal["nii", "neuroglancer/precomputed"] = None
    url: str = None
    space_id: str = None

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

    @staticmethod
    def NiiUrl(url: str) -> Union[nib.Nifti1Image, nib.Nifti2Image]:
        filename = CACHE.build_filename(url, suffix=".nii.gz")
        if not Path(filename).exists():
            with open(filename, "wb") as fp:
                resp = requests.get(url)
                resp.raise_for_status()
                fp.write(resp.content)
        nii = nib.load(filename)
        return nii

    @property
    def boundingbox(self):
        return Image._GetBBox(self)

    @property
    def data(self):
        assert self.format == "nii", "Can only get data of nii."
        return Image.NiiUrl(self.url)

    def plot(self, *args, **kwargs):
        raise NotImplementedError


@_loc_intersection.register(Pt, Image)
def compare_pt_to_image(pt: Pt, image: Image):
    ptcloud = PointCloud(space_id=pt.space_id, coordinates=[pt.coordinate])
    intersection = compare_ptcloud_to_image(ptcloud=ptcloud, image=image)
    if intersection:
        return pt


@_loc_intersection.register(PointCloud, Image)
def compare_ptcloud_to_image(ptcloud: PointCloud, image: Image):
    return intersect_ptcld_image(ptcloud=ptcloud, image=image)


def intersect_ptcld_image(ptcloud: PointCloud, image: Image) -> PointCloud:
    if image.space_id != ptcloud.space_id:
        raise InvalidAttrCompException(
            "ptcloud and image are in different space. Cannot compare the two."
        )

    value_outside = 0

    img = image.data
    arr = np.asanyarray(img.dataobj)

    # transform the points to the voxel space of the volume for extracting values
    phys2vox = np.linalg.inv(img.affine)
    voxels = PointCloud.transform(ptcloud, phys2vox)
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
