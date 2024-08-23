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

from dataclasses import dataclass, replace, asdict
from typing import Union, Tuple, List

import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from pathlib import Path
from hashlib import md5
from io import BytesIO
import gzip

from .base import VolumeProvider
from ...locations import point, pointcloud, BoundingBox
from ...locations.ops.intersection import _loc_intersection
from ....dataops.volume_fetcher.volume_fetcher import (
    get_volume_fetcher,
    get_bbox_getter,
    FetchKwargs,
    IMAGE_FORMATS,
    SIIBRA_MAX_FETCH_SIZE_GIB,
)


@dataclass
class ImageProvider(VolumeProvider):
    schema: str = "siibra/attr/data/image/v0.1"

    def __post_init__(self):
        assert (
            self.format in IMAGE_FORMATS
        ), f"Expected image format {self.format} to be in {IMAGE_FORMATS}, but was not."

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
            yield f"Image (format={self.format}): to zippable error.", ".error.txt", BytesIO(
                str(e).encode("utf-8")
            )

    def _points_to_voxels_coords(
        self, ptcloud: Union["point.Point", "pointcloud.PointCloud"], **fetch_kwargs
    ) -> Tuple[int, int, int]:
        if ptcloud.space_id != self.space_id:
            raise ValueError(
                "Points and Image must be in the same space. You can warp points "
                "space of the image with `warp()` method."
            )

        if isinstance(ptcloud, point.Point):
            ptcloud_ = pointcloud.PointCloud.from_points(points=[ptcloud])
        else:
            ptcloud_ = ptcloud

        nii = self.fetch(**fetch_kwargs)

        # transform the points to the voxel space of the volume for extracting values
        phys2vox = np.linalg.inv(nii.affine)
        voxels = pointcloud.PointCloud.transform(ptcloud_, phys2vox)
        x, y, z = np.round(voxels.coordinates).astype('int').T
        return x, y, z

    def lookup_points(
        self,
        points: Union["point.Point", "pointcloud.PointCloud"],
        **fetch_kwargs: FetchKwargs,
    ):
        """
        Evaluate the image at the positions of the given points.

        Note
        ----
        Uses nearest neighbor interpolation. Other interpolation schemes are not
        yet implemented.

        Parameters
        ----------
        ptcloud: PointSet
        outside_value: int, float. Default: 0
        fetch_kwargs: dict
            Any additional arguments are passed to the `fetch()` call for
            retrieving the image data.
        """
        x, y, z = self._points_to_voxels_coords(points, **fetch_kwargs)
        return self._read_voxels(x=x, y=y, z=z, **fetch_kwargs)

    def _read_voxels(
        self,
        x: Union[int, np.ndarray, List],
        y: Union[int, np.ndarray, List],
        z: Union[int, np.ndarray, List],
        **fetch_kwargs,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Read out the values of this Image for a given set of voxel coordinates.

        Note
        ----
        The fetch_kwargs are passed on to the `image.fetch()`.

        Parameters
        ----------
        voxel_coordinates : Union[ Tuple[int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray] ]


        Returns
        -------
        Iterator[valid_voxel_indices, values]
            First array provides the indicies of valid voxel coordinates, and the
            second contains the respective values. Invalid voxel coordinate means
            the values lie outside of the nifti array.
        """
        xyz = np.stack(
            [np.array([_]) if isinstance(_, int) else np.array(_) for _ in (x, y, z)],
            axis=1,
        )
        nii = self.fetch(**fetch_kwargs)
        valid_points_mask = np.all(
            [
                (0 <= dim_val) & (dim_val < dim_len)
                for dim_len, dim_val in zip(nii.shape, xyz.T)
            ],
            axis=0,
        )
        valid_x, valid_y, valid_z = xyz[valid_points_mask].T
        valid_points_indices, *_ = np.where(valid_points_mask)
        nii_arr = np.asanyarray(nii.dataobj).astype(nii.dataobj.dtype)
        valid_nii_values = nii_arr[valid_x, valid_y, valid_z]
        return valid_points_indices.astype(int).tolist(), valid_nii_values

    def get_intersection_scores(
        self,
        item: Union[point.Point, pointcloud.PointCloud, BoundingBox, "ImageProvider"],
        iou_lower_threshold: Union[int, float] = 0.0,
        voxel_sigma_threshold: int = 3,
        statistical_map_lower_threshold: float = 0.0,
        split_components: bool = False,
        **fetch_kwargs: FetchKwargs,
    ):
        from pandas import DataFrame
        from .ops.assignment import get_intersection_scores

        assignments = get_intersection_scores(
            queryitem=item,
            target_image=self,
            split_components=split_components,
            voxel_sigma_threshold=voxel_sigma_threshold,
            iou_lower_threshold=iou_lower_threshold,
            target_masking_lower_threshold=statistical_map_lower_threshold,
            **fetch_kwargs,
        )

        assignments_unpacked = [asdict(a) for a in assignments]

        return (
            DataFrame(assignments_unpacked)
            .convert_dtypes()  # convert will guess numeric column types
            .dropna(axis="columns", how="all")
        )


def from_nifti(nifti: Union[str, nib.Nifti1Image], space_id: str, **kwargs) -> "ImageProvider":
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
    return ImageProvider(format="nii", url=filename, space_id=space_id, **kwargs)


def colorize(
    image: ImageProvider, value_mapping: dict, **fetch_kwargs: FetchKwargs
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


@_loc_intersection.register(point.Point, ImageProvider)
def compare_pt_to_image(pt: point.Point, image: ImageProvider):
    ptcloud = pointcloud.PointCloud(space_id=pt.space_id, coordinates=[pt.coordinate])
    intersection = compare_ptcloud_to_image(ptcloud=ptcloud, image=image)
    if intersection:
        return pt


@_loc_intersection.register(pointcloud.PointCloud, ImageProvider)
def compare_ptcloud_to_image(ptcloud: pointcloud.PointCloud, image: ImageProvider):
    return intersect_ptcld_image(ptcloud=ptcloud, image=image)


def intersect_ptcld_image(
    ptcloud: pointcloud.PointCloud, image: ImageProvider
) -> pointcloud.PointCloud:
    value_outside = 0
    values = image.lookup_points(ptcloud)
    inside = list(np.where(values != value_outside)[0])
    return replace(
        ptcloud,
        coordinates=[ptcloud.coordinates[i] for i in inside],
        sigma=[ptcloud.sigma[i] for i in inside],
    )


@_loc_intersection.register(ImageProvider, ImageProvider)
def intersect_image_to_image(image0: ImageProvider, image1: ImageProvider):
    nii0 = image0.fetch()
    nii1 = image1.fetch()
    if np.issubdtype(nii0.dataobj, np.floating) or np.issubdtype(
        nii1.dataobj, np.floating
    ):
        pass
    else:
        elementwise_mask_intersection = intersect_nii_to_nii(nii0, nii1)
        return from_nifti(
            elementwise_mask_intersection,
            space_id=image0.space_id,
        )


def resample_img_to_img(
    source_img: nib.Nifti1Image, target_img: nib.Nifti1Image, interpolation: str = ""
) -> nib.Nifti1Image:
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


def intersect_nii_to_nii(nii0: nib.Nifti1Image, nii1: nib.Nifti1Image):
    """
    Get the intersection of the images' masks.

    Note
    ----
    Assumes the background is 0 for both nifti.

    Parameters
    ----------
    nii0 : nib.Nifti1Image
    nii1 : nib.Nifti1Image

    Returns
    -------
    nib.Nifti1Image
        returns a mask (i.e. dtype('uint8'))
    """
    arr0 = np.asanyarray(nii0.dataobj).astype("uint8")
    nii1_on_nii0 = resample_img_to_img(nii1, nii0)
    arr1 = np.asanyarray(nii1_on_nii0.dataobj).astype("uint8")
    elementwise_min = np.minimum(arr0, arr1)
    return nib.Nifti1Image(dataobj=elementwise_min, affine=nii0.affine)
