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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Callable, List
import requests

import numpy as np
import nibabel as nib
from neuroglancer_scripts.http_accessor import HttpAccessor
from neuroglancer_scripts.precomputed_io import (
    get_IO_for_existing_dataset,
    PrecomputedIO,
)

from ...volume_fetcher.volume_fetcher import register_volume_fetcher, FetchKwargs
from ....cache import fn_call_cache
from ....attributes.locations import BBox
from ....commons_new.logger import logger

from ....commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ....attributes.dataitems import Image


def extract_label_mask(arr: np.ndarray, label: int):
    return np.asanyarray(arr == label, dtype='uint8')


@fn_call_cache
def get_info(url: str) -> Dict:
    return requests.get(f"{url}/info").json()


@fn_call_cache
def get_transform_nm(url: str) -> Dict:
    return requests.get(f"{url}/transform.json").json()


def get_io(url: str) -> PrecomputedIO:
    accessor = HttpAccessor(url)
    return get_IO_for_existing_dataset(accessor)


def get_dtype(url: str):
    info = get_info(url)
    return np.dtype(info["data_type"])


@dataclass(init=False, unsafe_hash=True, eq=False)
class Scale:
    url: str  # TODO: consider what is better: pass the url or pass a NG class that holds the url
    chunk_sizes: np.ndarray
    encoding: str
    key: str
    resolution_nanometer: np.ndarray
    size: int
    voxel_offset: np.ndarray

    def __init__(self, scaleinfo: dict, url: str):
        self.url = url
        self.chunk_sizes = np.array(scaleinfo["chunk_sizes"]).squeeze()
        self.encoding = scaleinfo["encoding"]
        self.key = scaleinfo["key"]
        self.resolution_nanometer = np.array(scaleinfo["resolution"]).squeeze()
        self.size = scaleinfo["size"]
        self.voxel_offset = np.array(scaleinfo["voxel_offset"])

    def __eq__(self, other):
        assert isinstance(other, Scale)
        return (self.url == other.url) and np.array_equal(
            self.resolution_mm, other.resolution_mm
        )

    def __lt__(self, other):
        """Sort scales by resolution."""
        if not isinstance(other, Scale):
            raise ValueError(f"Cannot compare a Neuroglancer Scale with {type(other)}")
        return all(
            self.resolution_nanometer[i] < other.resolution_nanometer[i]
            for i in range(3)
        )

    def read_chunk(self, gx, gy, gz, channel: int = None):
        if any(g < 0 for g in (gx, gy, gz)):
            raise RuntimeError(
                "Negative tile index observed - you have likely requested fetch() "
                "with a bbox specification ranging outside the actual data."
            )

        x0 = gx * self.chunk_sizes[0]
        y0 = gy * self.chunk_sizes[1]
        z0 = gz * self.chunk_sizes[2]
        x1, y1, z1 = np.minimum(self.chunk_sizes + [x0, y0, z0], self.size)
        chunk_czyx = get_io(self.url).read_chunk(self.key, (x0, x1, y0, y1, z0, z1))
        if channel is None:
            channel = 0
        if channel + 1 > chunk_czyx.shape[0]:
            raise ValueError(f"There are only {chunk_czyx.shape[0]} color channels.")
        chunk_zyx = chunk_czyx[channel]
        return chunk_zyx

    @property
    def resolution_mm(self):
        return self.resolution_nanometer / 1e6

    @property
    def affine(self):
        scaling = np.diag(np.r_[self.resolution_nanometer, 1.0])
        affine = np.dot(get_transform_nm(self.url), scaling)
        affine[:3, :] /= 1e6
        return affine

    def validate_resolution(self, resolution_mm: float):
        """Test whether the resolution of this scale is sufficient to provide the given resolution."""
        return all(r <= resolution_mm for r in self.resolution_mm)

    def _estimate_nbytes(self, bbox: "BBox" = None):
        """Estimate the size image array to be fetched in bytes, given a bounding box."""
        if bbox is None:
            bbox_ = BBox(minpoint=(0, 0, 0), maxpoint=self.size, space_id=None)
        else:
            bbox_ = BBox.transform(bbox, np.linalg.inv(self.affine))
        result = get_dtype(self.url).itemsize * bbox_.volume
        logger.debug(
            f"Approximate size for fetching resolution "
            f"({', '.join(map('{:.6f}'.format, self.resolution_mm))}) mm "
            f"is {result / 1024**3:.5f} GiB."
        )
        return result

    def _point_to_integral_chunk_idx(self, xyz, integral_func: Callable):
        return (
            integral_func((np.array(xyz) - self.voxel_offset) / self.chunk_sizes)
            .astype("int")
            .ravel()
        )

    def fetch(self, bbox=None, channel: int = None, label: int = None):
        # define the bounding box in this scale's voxel space
        if bbox is None:
            bbox_ = BBox(minpoint=(0, 0, 0), maxpoint=self.size, space_id=None)
        else:
            bbox_ = bbox.transform(np.linalg.inv(self.affine))

        for dim in range(3):
            if bbox_.shape[dim] < 1:
                # logger.warning(
                #     f"Bounding box in voxel space will be enlarged to voxel size 1 along axis {dim}."
                # )
                bbox_._maxpoint[dim] = bbox_._maxpoint[dim] + 1

        # extract minimum and maximum the chunk indices to be loaded
        gx0, gy0, gz0 = self._point_to_integral_chunk_idx(
            tuple(bbox_._minpoint), np.floor
        )
        gx1, gy1, gz1 = self._point_to_integral_chunk_idx(
            tuple(bbox_._maxpoint), np.ceil
        )

        # create requested data volume, and fill it with the required chunk data
        shape_zyx = np.array([gz1 - gz0, gy1 - gy0, gx1 - gx0]) * self.chunk_sizes[::-1]
        data_zyx = np.zeros(shape_zyx, dtype=get_dtype(self.url))
        for gx in range(gx0, gx1):
            x0 = (gx - gx0) * self.chunk_sizes[0]
            for gy in range(gy0, gy1):
                y0 = (gy - gy0) * self.chunk_sizes[1]
                for gz in range(gz0, gz1):
                    z0 = (gz - gz0) * self.chunk_sizes[2]
                    chunk = self.read_chunk(gx, gy, gz, channel)
                    z1, y1, x1 = np.array([z0, y0, x0]) + chunk.shape
                    data_zyx[z0:z1, y0:y1, x0:x1] = chunk

        # determine the remaining offset from the "chunk mosaic" to the
        # exact bounding box requested, to cut off undesired borders
        data_min = np.array([gx0, gy0, gz0]) * self.chunk_sizes
        x0, y0, z0 = (np.array(tuple(bbox_._minpoint)) - data_min).astype("int")
        xd, yd, zd = np.array(bbox_.shape).astype("int")
        offset = tuple(bbox_._minpoint)

        # build the nifti image
        trans = np.identity(4)[[2, 1, 0, 3], :]  # zyx -> xyz
        shift = np.c_[np.identity(4)[:, :3], np.r_[offset, 1]]

        arr = data_zyx[z0 : z0 + zd, y0 : y0 + yd, x0 : x0 + xd]
        if label is not None:
            arr = extract_label_mask(arr, label)
        return nib.Nifti1Image(
            arr,
            np.dot(self.affine, np.dot(shift, trans)),
        )


def get_scales(url: str) -> List["Scale"]:
    info = get_info(url)
    return sorted([Scale(scaleinfo, url=url) for scaleinfo in info["scales"]])


def get_affine(url: str):
    """The affine matrix of the first resolution scale."""
    return get_scales(url)[0].affine


def select_scale(
    scales: List[Scale],
    resolution_mm: float = None,
    bbox=None,
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
) -> Scale:
    max_bytes = max_download_GB * 1024**3

    if resolution_mm is None:  # no requirements, return lowest available resolution
        selected_scale = sorted(scales)[-1]
        assert selected_scale._estimate_nbytes(bbox) <= max_bytes
        return selected_scale

    reverse = (
        True if resolution_mm > max([s.resolution_mm.max() for s in scales]) else False
    )
    scales_ = sorted(scales, reverse=reverse)

    if resolution_mm == -1:  # highest possible is requested
        suitable_scales = [scales_[0]]
    else:  # closest to resolution_mm
        suitable_scales = [s for s in scales_ if s.validate_resolution(resolution_mm)]

    for scale in scales_:
        estimated_nbytes = scale._estimate_nbytes(bbox)
        if estimated_nbytes <= max_bytes:
            selected_scale = scale
            if selected_scale not in suitable_scales:
                logger.warning(
                    f"Resolution was reduced to {selected_scale.resolution_mm} to provide a feasible volume size of "
                    f"{max_download_GB} GiB. Set a higher `max_download_GB` to fetch in the requested resolution."
                )
            break
    else:
        raise RuntimeError(
            f"Cannot fetch {f'bounding box {bbox} ' if bbox else ''}since the lowest download size "
            f"{scale._estimate_nbytes(bbox) / 1024**3}GiB > {max_download_GB=}GiB."
        )

    return selected_scale


@register_volume_fetcher("neuroglancer/precomputed", "image")
def fetch_neuroglancer(image: "Image", fetchkwargs: FetchKwargs) -> "nib.Nifti1Image":
    scales = get_scales(image.url)
    scale = select_scale(
        scales,
        resolution_mm=fetchkwargs["resolution_mm"],
        bbox=fetchkwargs["bbox"],
        max_download_GB=fetchkwargs["max_download_GB"],
    )
    mapping = fetchkwargs["mapping"]
    if mapping is not None and len(mapping) == 1:
        details = next(iter(mapping.values()))
        return scale.fetch(bbox=fetchkwargs["bbox"], label=details.get("label"))
    else:
        return scale.fetch(bbox=fetchkwargs["bbox"])


@fn_call_cache
def _GetBBox(image: "Image"):
    # if image.format == "neuroglancer/precomputed":
    #     resp = requests.get(f"{image.url}/info")
    #     resp.raise_for_status()
    #     info_json = resp.json()

    #     resp = requests.get(f"{image.url}/transform.json")
    #     resp.raise_for_status()
    #     transform_json = resp.json()

    #     scale, *_ = info_json.get("scales")
    #     size = scale.get("size")
    #     resolution = scale.get("resolution")
    #     dimension = [s * r for s, r in zip(size, resolution)]
    #     xs, ys, zs = zip([0, 0, 0], dimension)
    #     corners = list(product(xs, ys, zs))
    #     hom = np.c_[corners, np.ones(len(corners))]
    #     new_coord = np.dot(np.array(transform_json), hom.T)[:3, :].T / 1e6

    #     min = np.min(new_coord, axis=0)
    #     max = np.max(new_coord, axis=0)
    #     return boundingbox.BBox(
    #         minpoint=min.tolist(), maxpoint=max.tolist(), space_id=image.space_id
    #     )
    raise NotImplementedError
