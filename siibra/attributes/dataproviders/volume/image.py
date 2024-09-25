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
from typing import Union, Tuple, List, Dict
from pathlib import Path
from hashlib import md5
from io import BytesIO
import gzip

import numpy as np
import nibabel as nib

from .base import VolumeProvider, VolumeOpsKwargs
from ...locations import point, pointcloud, BoundingBox
from ...locations.ops.intersection import _loc_intersection
from ....operations.volume_fetcher.nifti import NiftiExtractVOI
from ....operations.volume_fetcher.neuroglancer_precomputed import NgPrecomputedFetchCfg


@dataclass
class ImageProvider(VolumeProvider):
    schema: str = "siibra/attr/data/image/v0.1"

    @property
    def boundingbox(self) -> "BoundingBox":
        raise NotImplementedError

    def get_affine(self, **volume_ops_kwargs: VolumeOpsKwargs):
        # TODO: pull from source without fetching the whole image
        return self.get_data(**volume_ops_kwargs).affine

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def _iter_zippable(self):
        yield from super()._iter_zippable()
        try:
            nii = self.get_data()
            suffix = None
            if isinstance(nii, (nib.Nifti1Image, nib.Nifti2Image)):
                suffix = ".nii.gz"
            if isinstance(nii, nib.GiftiImage):
                suffix = ".gii.gz"
            if not suffix:
                raise RuntimeError(
                    "Image.get_data returning a non nifti, non gifti image"
                )

            # TODO not ideal, since loads everything in memory. Ideally we can stream it as IO
            gzipped = gzip.compress(nii.to_bytes())
            yield f"Image (format={self.format})", suffix, BytesIO(gzipped)
        except Exception as e:
            yield f"Image (format={self.format}): to zippable error.", ".error.txt", BytesIO(
                str(e).encode("utf-8")
            )

    def _points_to_voxels_coords(
        self,
        ptcloud: Union["point.Point", "pointcloud.PointCloud"],
        **volume_ops_kwargs,
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

        nii = self.get_data(**volume_ops_kwargs)

        # transform the points to the voxel space of the volume for extracting values
        phys2vox = np.linalg.inv(nii.affine)
        voxels = pointcloud.PointCloud.transform(ptcloud_, phys2vox)
        x, y, z = np.round(voxels.coordinates).astype("int").T
        return x, y, z

    def lookup_points(
        self,
        points: Union["point.Point", "pointcloud.PointCloud"],
        **volume_ops_kwargs: VolumeOpsKwargs,
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
        volume_ops_kwargs: dict
            Any additional arguments are passed to the `fetch()` call for
            retrieving the image data.
        """
        x, y, z = self._points_to_voxels_coords(points, **volume_ops_kwargs)
        return self._read_voxels(x=x, y=y, z=z, **volume_ops_kwargs)

    def _read_voxels(
        self,
        x: Union[int, np.ndarray, List],
        y: Union[int, np.ndarray, List],
        z: Union[int, np.ndarray, List],
        **volume_ops_kwargs,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Read out the values of this Image for a given set of voxel coordinates.

        Note
        ----
        The volume_ops_kwargs are passed on to the `image.get_data()`.

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
        nii = self.get_data(**volume_ops_kwargs)
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
        **volume_ops_kwargs: VolumeOpsKwargs,
    ):
        from pandas import DataFrame
        from ....operations.image_assignment import get_intersection_scores

        assignments = get_intersection_scores(
            queryitem=item,
            target_image=self,
            split_components=split_components,
            voxel_sigma_threshold=voxel_sigma_threshold,
            iou_lower_threshold=iou_lower_threshold,
            target_masking_lower_threshold=statistical_map_lower_threshold,
            **volume_ops_kwargs,
        )

        assignments_unpacked = [asdict(a) for a in assignments]

        return (
            DataFrame(assignments_unpacked)
            .convert_dtypes()  # convert will guess numeric column types
            .dropna(axis="columns", how="all")
        )

    def get_data(self, **kwargs) -> nib.Nifti1Image:
        return super().get_data(**kwargs)

    def query(
        self, *arg, bbox: "BoundingBox" = None, resolution_mm: float = None, **kwargs
    ):
        """
        Return a copy of the image provider with the following constrains (if provided):

        - bounded by bbox (works on all formats)
        - use closest resolution_mm (only on neuroglancer precomputed format)
        """
        new_img_prov = replace(self)
        if bbox is not None:
            new_img_prov.append_op(NiftiExtractVOI.generate_specs(voi=bbox))
        if resolution_mm is not None:
            new_img_prov.append_op(
                NgPrecomputedFetchCfg.generate_specs(
                    fetch_config={"resolution_mm": resolution_mm}
                )
            )
        return new_img_prov


def from_pointcloud(
    pointcloud: pointcloud.PointCloud,
    normalize=True,
    cached=False,
    target: ImageProvider = None,
) -> ImageProvider:
    from ....operations.base import Of
    from ....operations.volume_fetcher.nifti import NiftiFromPointCloud
    from ....operations.volume_fetcher.nifti import ResampleNifti

    transformation_ops = []
    if target is not None:
        transformation_ops = [
            ResampleNifti.generate_specs(target_img=target.get_data())
        ]
    return ImageProvider(
        format="nii",
        override_ops=[
            Of.generate_specs(instance=pointcloud, force=(not cached)),
            NiftiFromPointCloud.generate_specs(normalize=normalize, force=(not cached)),
            *transformation_ops,
        ],
        space_id=pointcloud.space_id,
    )


def from_nifti(
    nifti: Union[str, nib.Nifti1Image],
    space: str = None,
    space_id: str = None,
    **kwargs,
) -> "ImageProvider":
    """
    Builds an `Image` `Attribute` from a Nifti image or path to a nifti file.
    Use space_id kwargs if you want to directly set the space_id. Otherwise, use the space kwarg to specify a space to look up.
    space_id is ignored if space is provided
    """
    from ....cache import CACHE
    from .... import get_space

    if space is not None:
        space_id = get_space(space).ID

    filename = None
    if isinstance(nifti, str):
        filename = nifti
        if not nifti.startswith("http"):
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


def from_array(
    data: np.array,
    affine: np.array,
    space: str = None,
    space_id: str = None,
    **kwargs,
) -> "ImageProvider":
    """
    Builds an `Image` `Attribute` from a volumetric array and affine matrix.
    Use space_id kwargs if you want to directly set the space_id. Otherwise, use the space kwarg to specify a space to look up.
    space_id is ignored if space is provided
    """
    return from_nifti(
        nib.Nifti1Image(data, affine), space=space, space_id=space_id, **kwargs
    )


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
    inside = np.array(values[0])[np.where(values[1] != value_outside)[0]].tolist()
    return replace(
        ptcloud,
        coordinates=[ptcloud.coordinates[i] for i in inside],
        sigma=[ptcloud.sigma[i] for i in inside],
    )


@_loc_intersection.register(ImageProvider, ImageProvider)
def intersect_image_to_image(image0: ImageProvider, image1: ImageProvider):
    nii0 = image0.get_data()
    nii1 = image1.get_data()
    if np.issubdtype(nii0.dataobj, np.floating) or np.issubdtype(
        nii1.dataobj, np.floating
    ):
        pass
    else:
        raise NotImplementedError
        # TODO reimplement this asd a DataOp
        # elementwise_mask_intersection = intersect_nii_to_nii(nii0, nii1)
        # return from_nifti(
        #     elementwise_mask_intersection,
        #     space_id=image0.space_id,
        # )
