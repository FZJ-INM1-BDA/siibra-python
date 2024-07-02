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
from ....locations import BBox

from ....commons import logger

if TYPE_CHECKING:
    from ....dataitems import Image


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


@dataclass(init=False, unsafe_hash=True)
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

    def __lt__(self, other):
        """Sort scales by resolution."""
        if not isinstance(other, Scale):
            raise ValueError(f"Cannot compare a Neuroglancer Scale with {type(other)}")
        return all(self.resolution_nanometer[i] < other.resolution_nanometer[i] for i in range(3))

    def read_chunk(self, gx, gy, gz, channel: int = None):
        if any(g < 0 for g in (gx, gy, gz)):
            raise RuntimeError(
                "Negative tile index observed - you have likely requested fetch() with a voi specification ranging outside the actual data."
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
            bbox_ = bbox.transform(np.linalg.inv(self.affine))
        result = get_dtype(self.url).itemsize * bbox_.volume
        # logger.debug(
        #     f"Approximate size for fetching resolution "
        #     f"({', '.join(map('{:.6f}'.format, self.res_mm))}) mm "
        #     f"is {result / 1024**3:.5f} GiB."
        # )
        return result

    def _point_to_integral_chunk_idx(self, xyz, integral_func: Callable):
        return (
            integral_func((np.array(xyz) - self.voxel_offset) / self.chunk_sizes)
            .astype("int")
            .ravel()
        )

    def fetch(self, bbox=None, channel: int = None):
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
        return nib.Nifti1Image(
            data_zyx[z0 : z0 + zd, y0 : y0 + yd, x0 : x0 + xd],
            np.dot(self.affine, np.dot(shift, trans)),
        )


@fn_call_cache
def get_scales(url: str) -> List["Scale"]:
    info = get_info(url)
    return sorted([Scale(scaleinfo, url=url) for scaleinfo in info["scales"]])


# def get_shape(self):
# scale = self._select_scale(resolution_mm=resolution_mm, max_bytes=max_bytes)
#         return scale.size
#     # return the shape of the scale 0 array
#     return self.scales[0].size


# def affine(self):
#     # return the affine matrix of the scale 0 data
#     return self.scales[0].affine


# class NeuroglancerPrecomputed:

#     def __init__(self, url: str, info: Dict):
#         self.url = url

#     @property
#     def transform_nm(self):
#         return get_transform_nm(self.url)

#     @property
#     def dtype(self):
#         return get_dtype(self.url)


def select_scale(
    scales: List[Scale],
    resolution_mm: float,
    bbox=None,
    max_bytes: float = 0.2 * 1024**2,
) -> Scale:
    if resolution_mm is None:
        suitable = scales
    elif resolution_mm < 0:
        suitable = [scales[0]]
    else:
        suitable = sorted(s for s in scales if s.validate_resolution(resolution_mm))

    if len(suitable) > 0:
        scale = suitable[-1]
    else:
        scale = scales[0]
        xyz_res = ["{:.6f}".format(r).rstrip("0") for r in scale.resolution_mm]
        if all(r.startswith(str(resolution_mm)) for r in xyz_res):
            logger.info(f"Closest resolution to requested is {', '.join(xyz_res)} mm.")
        else:
            logger.warning(
                f"Requested resolution {resolution_mm} is not available. "
                f"Falling back to the highest possible resolution of "
                f"{', '.join(xyz_res)} mm."
            )

    for scale in scales:
        if scale._estimate_nbytes(bbox) > max_bytes:
            break
    else:
        raise RuntimeError(
            f"Fetching bounding box {bbox} is infeasible "
            f"relative to the limit of {max_bytes / 1024**3}GiB."
        )

    # if scale_changed:
    #     logger.warning(
    #         f"Resolution was reduced to {scale.resolution_mm} to provide a "
    #         f"feasible volume size of {max_bytes / 1024**3} GiB. Set `max_bytes` to"
    #         f" fetch in the resolution requested."
    #     )
    return scale


@register_volume_fetcher("neuroglancer/precomputed", "image")
def fetch_neuroglancer(image: "Image", fetchkwargs: FetchKwargs) -> "nib.Nifti1Image":
    scales = get_scales(image.url)
    scale = select_scale(
        scales, resolution_mm=fetchkwargs["resolution_mm"], bbox=fetchkwargs["bbox"]
    )
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
