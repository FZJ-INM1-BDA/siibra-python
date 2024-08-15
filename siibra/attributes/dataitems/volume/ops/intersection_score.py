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

from dataclasses import dataclass, asdict
from typing import Union, Tuple, Generator, List

import numpy as np
from nibabel import Nifti1Image

from ..image import Image, from_nifti
from ....locations import Point, BoundingBox, PointCloud
from .....commons.maps import resample_img_to_img, compute_centroid
from .....retrieval.volume_fetcher import FetchKwargs


@dataclass
class ImageAssignment:
    input_structure_index: int
    centroid: Union[Tuple[np.ndarray], Point]
    map_value: np.ndarray


@dataclass
class NiftiIntersectionScore:
    intersection_over_union: float
    intersection_over_first: float
    intersection_over_second: float
    correlation: float
    weighted_mean_of_first: float
    weighted_mean_of_second: float
    map_value_mean: float
    map_value_std: float


@dataclass
class ScoredImageAssignment(ImageAssignment, NiftiIntersectionScore):
    pass


def get_connected_components(
    nii: Nifti1Image,
    background: int = 0,
    connectivity: int = 2,
    threshold: float = 0.0,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Provide an iterator over connected components in the array. If the image
    data is float (such as probability maps), it will convert to a mask and
    then find the connected components.

    Note
    ----
    `Uses skimage.measure.label()` to determine foreground compenents.

    Parameters
    ----------
    imgdata : np.ndarray
    background_value : int, Default: 0
    connectivity : int, Default: 2
    threshold: float, Default: 0.0
        The threshold used to create mask from probability maps, i.e, anything
        below set to 0 and rest to 1.

    Yields
    ------
    Generator[Tuple[int, np.ndarray], None, None]
        tuple of integer label of the component and component as an nd.array in
        the shape of the original image.
    """
    from skimage.measure import label as measure_label

    imgdata = np.asarray(nii.dataobj, dtype=nii.dataobj.dtype)
    mask = (imgdata > threshold).astype("uint8")
    components = measure_label(mask, connectivity=connectivity, background=background)
    component_labels = np.unique(components)
    yield from (
        (
            label,
            Nifti1Image(
                dataobj=imgdata * (components == label).astype("uint8"),
                affine=nii.affine,
            ),
        )
        for label in component_labels
        if label > 0
    )


def pearson_correlation_coefficient(arr1: np.ndarray, arr2: np.ndarray):
    """
    See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Parameters
    ----------
    arr1 : np.ndarray
        _description_
    arr2 : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """
    a1_0 = arr1 - arr1.mean()
    a2_0 = arr2 - arr2.mean()
    dem = np.sqrt(np.sum(a1_0**2) * np.sum(a2_0**2))
    if dem == 0:
        return 0
    else:
        return np.sum(np.multiply(a1_0, a2_0)) / dem


def calculate_nifti_intersection_score(
    nii1: Nifti1Image, nii2: Nifti1Image, target_masking_lower_threshold: float = 0.0
):
    """
    Compare two arrays in physical space as defined by the given affine matrices.
    Matrices map voxel coordinates to physical coordinates.
    This function uses the object id to cache extraction of the nonzero coordinates.
    Repeated calls involving the same map will therefore be much faster as they
    will only access the image array if overlapping pixels are detected.

    It is recommended to install the indexed-gzip package,
    which will further speed this up.
    """
    arr1 = np.asanyarray(nii1.dataobj).squeeze()
    arr2 = np.asanyarray(nii2.dataobj).squeeze()

    def homog(XYZ):
        return np.c_[XYZ, np.ones(XYZ.shape[0])]

    def colsplit(XYZ):
        return np.split(XYZ, 3, axis=1)

    # Compute the nonzero voxels in map2 and their correspondences in map1
    nz_voxels_nii2 = np.c_[np.nonzero(arr2 > 0)]
    warp2on1 = np.dot(np.linalg.inv(nii1.affine), nii2.affine)
    nz_voxels_nii2_warped_to_nii1 = (
        np.dot(warp2on1, homog(nz_voxels_nii2).T).T[:, :3]
    ).astype("int")

    # valid voxel pairs
    valid = np.all(
        np.logical_and.reduce(
            [
                nz_voxels_nii2_warped_to_nii1 >= 0,
                nz_voxels_nii2_warped_to_nii1 < arr1.shape[:3],
                nz_voxels_nii2 >= 0,
                nz_voxels_nii2 < arr2.shape[:3],
            ]
        ),
        1,
    )
    X1, Y1, Z1 = colsplit(nz_voxels_nii2_warped_to_nii1[valid, :])
    X2, Y2, Z2 = colsplit(nz_voxels_nii2[valid, :])

    # intersection
    v1, v2 = arr1[X1, Y1, Z1].squeeze(), arr2[X2, Y2, Z2].squeeze()
    m1, m2 = ((v > 0).astype("uint8") for v in [v1, v2])
    intersection = np.minimum(m1, m2).sum()
    if intersection == 0:
        return NiftiIntersectionScore(
            intersection_over_union=0,
            intersection_over_first=0,
            intersection_over_second=0,
            correlation=0,
            weighted_mean_of_first=0,
            weighted_mean_of_second=0,
            map_value_mean=0,
            map_value_std=0,
        )

    # Compute the nonzero voxels in map1 with their correspondences in map2
    nz_voxels_nii1 = np.c_[np.nonzero(arr1 > 0)]
    warp1on2 = np.dot(np.linalg.inv(nii2.affine), nii1.affine)

    # Voxels referring to the union of the nonzero pixels in both maps
    XYZa1 = np.unique(
        np.concatenate((nz_voxels_nii1, nz_voxels_nii2_warped_to_nii1)), axis=0
    )
    XYZa2 = (np.dot(warp1on2, homog(XYZa1).T).T[:, :3]).astype("int")
    valid = np.all(
        np.logical_and.reduce(
            [XYZa1 >= 0, XYZa1 < arr1.shape[:3], XYZa2 >= 0, XYZa2 < arr2.shape[:3]]
        ),
        1,
    )
    Xa1, Ya1, Za1 = colsplit(XYZa1[valid, :])
    Xa2, Ya2, Za2 = colsplit(XYZa2[valid, :])

    # pearson's r wrt to full size image
    a1 = arr1[Xa1, Ya1, Za1].squeeze()
    a2 = arr2[Xa2, Ya2, Za2].squeeze()
    rho = pearson_correlation_coefficient(a1, a2)

    union = np.maximum((arr1 > 0).astype("uint8"), (arr2 > 0).astype("uint8")).sum()

    return NiftiIntersectionScore(
        intersection_over_union=intersection / union,
        intersection_over_first=intersection / (a1 > 0).sum(),
        intersection_over_second=intersection / (a2 > 0).sum(),
        correlation=rho,
        weighted_mean_of_first=np.sum(a1 * a2) / np.sum(a2),
        weighted_mean_of_second=np.sum(a1 * a2) / np.sum(a1),
        map_value_mean=a1.mean(),
        map_value_std=a1.std(),
    )


def get_image_intersection_score(
    query_image: Image,
    target_image: Image,
    split_components: bool = False,
    iou_lower_threshold: float = 0.0,
    target_masking_lower_threshold: float = 0.0,
    **fetch_kwargs: FetchKwargs,
) -> List[ScoredImageAssignment]:
    # TODO: well-define thresholds and ensure where to use a mask or not
    assert query_image.space == target_image.space, ValueError(
        "Assigned volume must be in the same space as the map."
    )

    if split_components:
        iter_components = lambda arr: get_connected_components(arr)
    else:
        iter_components = lambda arr: [(0, arr)]

    querynii = query_image.fetch(**fetch_kwargs)
    target_nii = target_image.fetch(**fetch_kwargs)
    querynii_resamp = resample_img_to_img(querynii, target_nii)

    assignments: List[ScoredImageAssignment] = []
    for component_index, querynii_component in iter_components(querynii_resamp):
        score = calculate_nifti_intersection_score(
            querynii_component,
            target_nii,
            target_masking_lower_threshold=target_masking_lower_threshold,
        )
        if score.intersection_over_union <= iou_lower_threshold:
            continue
        assignments.append(
            ScoredImageAssignment(
                input_structure_index=component_index,
                centroid=compute_centroid(querynii_component).coordinate,
                map_value=None,
                **asdict(score),
            )
        )

    return assignments


def get_bounding_intersection_score(
    bbox: BoundingBox,
    image: Image,
    **fetch_kwargs: FetchKwargs,
):
    # quick check
    if image.boundingbox.intersect(bbox) is None:
        return []

    # assignments: List[ImageAssignment] = []

    raise NotImplementedError


def get_pointcloud_intersection_score(
    points: PointCloud,
    image: Image,
    voxel_sigma_threshold: int = 3,
    iou_lower_threshold: float = 0.0,
    target_masking_lower_threshold: float = 0.0,
    **fetch_kwargs: FetchKwargs,
):
    assignments: List[Union[ImageAssignment, ScoredImageAssignment]] = []

    points_wrpd = points.warp(
        image.space_id
    )  # returns the same points if in the same space

    for pointindex, pt in enumerate(points_wrpd):
        # build an Image of the Gaussian kernel, then recurse this into assign_image
        gaussian_kernel = pt.create_gaussian_kernel(
            image.get_affine(**fetch_kwargs), voxel_sigma_threshold
        )
        kernel_assignments = get_image_intersection_score(
            query_image=from_nifti(gaussian_kernel, image.space_id),
            target_image=image,
            split_components=False,
            iou_lower_threshold=iou_lower_threshold,
            target_masking_lower_threshold=target_masking_lower_threshold,
            **fetch_kwargs,
        )
        for score in kernel_assignments:
            if score.intersection_over_union <= iou_lower_threshold:
                continue
            score.input_structure_index = pointindex
            score.centroid = points[pointindex].coordinate
            assignments.append(score)

    return assignments


def get_intersection_scores(
    queryitem: Union[Point, PointCloud, BoundingBox, Image],
    target_image: Image,
    iou_lower_threshold: Union[int, float] = 0.0,
    target_masking_lower_threshold: float = 0.0,
    split_components: bool = False,
    **fetch_kwargs: FetchKwargs,
) -> List[Union[ImageAssignment, ScoredImageAssignment]]:
    if isinstance(queryitem, (Point, PointCloud)):
        pointcld = (
            queryitem
            if isinstance(queryitem, PointCloud)
            else PointCloud.from_points([queryitem])
        )
        return get_pointcloud_intersection_score(
            pointcld,
            image=target_image,
            iou_lower_threshold=iou_lower_threshold,
            target_masking_lower_threshold=target_masking_lower_threshold,
            **fetch_kwargs,
        )

    if isinstance(queryitem, BoundingBox):
        return get_bounding_intersection_score(
            bbox=queryitem,
            target_image=target_image,
            iou_lower_threshold=iou_lower_threshold,
            statistical_map_lower_threshold=target_masking_lower_threshold,
            **fetch_kwargs,
        )

    if isinstance(queryitem, Image):
        return get_image_intersection_score(
            query_image=queryitem,
            target_image=target_image,
            split_components=split_components,
            iou_lower_threshold=iou_lower_threshold,
            target_masking_lower_threshold=target_masking_lower_threshold,
            **fetch_kwargs,
        )

    raise TypeError(
        f"Items of type {queryitem.__class__.__name__} cannot be used for image "
        "intersection score calculation."
    )
