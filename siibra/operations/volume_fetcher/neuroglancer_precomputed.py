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
from typing import TYPE_CHECKING, Dict, Callable, List, Union, ClassVar, Tuple
import requests
from itertools import product
import numpy as np
import nibabel as nib
from neuroglancer_scripts.http_accessor import HttpAccessor
from neuroglancer_scripts.precomputed_io import (
    get_IO_for_existing_dataset,
    PrecomputedIO,
)
from concurrent.futures import ThreadPoolExecutor
from itertools import product


try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

from .base import PostProcVolProvider, VolumeFormats, DataOp
from ...cache import fn_call_cache
from ...commons.logger import logger
from ...commons.conf import SiibraConf

if TYPE_CHECKING:
    from ...attributes.dataproviders.volume import VolumeProvider
    from ...attributes.locations import BoundingBox

NG_VOLUME_FORMAT_STR = "neuroglancer/precomputed"


class Mapping(TypedDict):
    """
    Represents restrictions to apply to an image to get partial information,
    such as labelled mask, a specific slice etc.
    """

    label: int = None
    range: Tuple[float, float]
    subspace: Tuple[slice, ...]
    target: str = None


class VolumeOpsKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.

    Note
    ----
    Not all parameters are avaialble for all formats and volumes.
    """

    url: str = None
    bbox: "BoundingBox" = None
    resolution_mm: float = None
    max_download_GB: float = SiibraConf.SIIBRA_MAX_FETCH_SIZE_GIB
    mapping: Dict[str, Mapping] = None


def extract_label_mask(arr: np.ndarray, label: int):
    return np.asanyarray(arr == label, dtype="uint8")


@fn_call_cache
def get_info(url: str) -> Dict:
    resp = Scale._session.get(f"{url}/info")
    resp.raise_for_status()
    return resp.json()


@fn_call_cache
def get_transform_nm(url: str) -> Dict:
    resp = Scale._session.get(f"{url}/transform.json")
    resp.raise_for_status()
    return resp.json()


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
    size: List[int]
    voxel_offset: np.ndarray

    _session: ClassVar[requests.Session] = requests.Session()

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

    def _estimate_nbytes(self, bbox: Union["BoundingBox", None] = None):
        """Estimate the size image array to be fetched in bytes, given a bounding box."""
        from ...attributes.locations import BoundingBox

        if bbox is None:
            bbox_ = BoundingBox(minpoint=[0, 0, 0], maxpoint=self.size, space_id=None)
        else:
            bbox_ = BoundingBox.transform(bbox, np.linalg.inv(self.affine))
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

    def fetch(
        self,
        bbox: Union["BoundingBox", None] = None,
        channel: int = None,
        label: int = None,
    ):
        # define the bounding box in this scale's voxel space
        from ...attributes.locations import BoundingBox

        if bbox is None:
            bbox_ = BoundingBox(minpoint=[0, 0, 0], maxpoint=self.size, space_id=None)
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

        # Since the bottlenet is network, use a ThreadPoolExecutor
        # cuts op from 108.29645109176636 sec -> 12.598513841629028 sec
        # with same md5sum

        def _read_chunk(gxyz):
            gx, gy, gz = gxyz
            chunk = self.read_chunk(gx, gy, gz, channel)
            x0 = (gx - gx0) * self.chunk_sizes[0]
            y0 = (gy - gy0) * self.chunk_sizes[1]
            z0 = (gz - gz0) * self.chunk_sizes[2]
            return chunk, x0, y0, z0

        with ThreadPoolExecutor() as ex:
            for chunk, x0, y0, z0 in ex.map(
                _read_chunk,
                product(
                    range(gx0, gx1),
                    range(gy0, gy1),
                    range(gz0, gz1),
                ),
            ):
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
    resolution_mm: Union[None, float] = None,
    bbox=None,
    max_download_GB: float = SiibraConf.SIIBRA_MAX_FETCH_SIZE_GIB,
) -> Scale:
    max_bytes = max_download_GB * 1024**3

    # TODO if resolution_mm is unset, should we select the scale with the highest resolution
    # but still below max_download_GB?
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
            f"{scale._estimate_nbytes(bbox) / 1024**3}GiB > max_download_GB={max_download_GB}GiB."
        )
    return selected_scale


def fetch_neuroglancer(url: str, **fetchkwargs: VolumeOpsKwargs) -> "nib.Nifti1Image":
    scales = get_scales(url)
    scale = select_scale(
        scales,
        resolution_mm=fetchkwargs.get("resolution_mm"),
        bbox=fetchkwargs.get("bbox"),
        max_download_GB=fetchkwargs.get(
            "max_download_GB", SiibraConf.SIIBRA_MAX_FETCH_SIZE_GIB
        ),
    )
    return scale.fetch(bbox=fetchkwargs.get("bbox"))


@VolumeFormats.register_format_read(NG_VOLUME_FORMAT_STR, "image")
class NgVolPostProcImgProvider(PostProcVolProvider):

    @classmethod
    def _verify_image_provider(cls, volume_provider: "VolumeProvider"):
        assert (
            volume_provider.format == NG_VOLUME_FORMAT_STR
        ), f"Expected format to be {NG_VOLUME_FORMAT_STR}, but was {volume_provider.format}"

    @classmethod
    def on_post_init(cls, volume_provider: "VolumeProvider"):
        cls._verify_image_provider(volume_provider)

        try:
            assert volume_provider.ops[-1]["type"] == ReadNeuroglancerPrecomputed.type
        except (IndexError, AssertionError):
            fetch_kwarg_to_nifti_op = ReadNeuroglancerPrecomputed.generate_specs()
            if len(volume_provider.override_ops) > 0:
                volume_provider.override_ops.append(fetch_kwarg_to_nifti_op)
            else:
                volume_provider.transformation_ops.append(fetch_kwarg_to_nifti_op)

    @classmethod
    def on_append_op(cls, volume_provider: "VolumeProvider", op: Dict):
        from .nifti import NiftiExtractVOI
        from ...attributes.dataproviders.volume import VolumeProvider

        cls._verify_image_provider(volume_provider)

        if volume_provider.ops[-1]["type"] != ReadNeuroglancerPrecomputed.type:
            raise RuntimeError(
                f"neuroglancer volume must have ReadNeuroglancerPrecomputed as the final op. {volume_provider.ops[-1]['type']} was found."
            )

        fetch_op = volume_provider.pop_op()
        if op["type"] == NiftiExtractVOI.type:
            bbox = op["voi"]
            op = NgPrecomputedFetchCfg.generate_specs(fetch_config={"bbox": bbox})
        super(VolumeProvider, volume_provider).append_op(op)

        if len(volume_provider.override_ops) > 0:
            volume_provider.override_ops.append(fetch_op)
        else:
            volume_provider.transformation_ops.append(fetch_op)

    @classmethod
    def on_get_retrieval_ops(cls, volume_provider: "VolumeProvider"):
        cls._verify_image_provider(volume_provider)
        return [
            NgPrecomputedFetchCfg.generate_specs(
                fetch_config={"url": volume_provider.url}
            )
        ]


class ReadNeuroglancerPrecomputed(DataOp):
    input: Union[None, VolumeOpsKwargs]
    output: nib.Nifti1Image
    desc = "Directly read neuroglancer volume"
    type = "read/neuroglancer_precomputed"

    def run(self, input, url: str = None, **kwargs):
        if input is None:
            input = {}
        assert isinstance(input, dict)

        kwargs = {
            **input,
            **kwargs,
        }

        kwarg_url = kwargs.pop("url", None)
        if kwarg_url and url:
            logger.warning(
                f"url is provided both in kwarg {kwarg_url}, as well as positional arg {url}. Ignoring kwarg url"
            )
        url = url or kwarg_url
        assert url is not None
        print(url, kwargs)
        return fetch_neuroglancer(url, **kwargs)

    @classmethod
    def generate_specs(cls, *, url: str = None, **kwargs: VolumeOpsKwargs):
        base = super().generate_specs(**kwargs)
        return {**base, "url": url}


class NgPrecomputedFetchCfg(DataOp):
    input: Union[None, VolumeOpsKwargs]
    output: VolumeOpsKwargs
    desc = "Creating/Updating neuroglancer fetch config"
    type = "volume/ngprecomp/update_cfg"

    def run(self, input, url, fetch_config: VolumeOpsKwargs, **kwargs):
        return {
            **(input or {}),
            **fetch_config,
        }

    @classmethod
    def generate_specs(
        cls, url: str = None, fetch_config: VolumeOpsKwargs = None, **kwargs
    ):
        base = super().generate_specs(**kwargs)
        return {**base, "url": url, "fetch_config": fetch_config or {}}


@fn_call_cache
def fetch_ng_bbox(
    image: "VolumeProvider", fetchkwargs: Union[VolumeOpsKwargs, None] = None
) -> "BoundingBox":
    assert image.format == NG_VOLUME_FORMAT_STR
    from ...attributes.locations import BoundingBox

    provided_bbox = fetchkwargs["bbox"] if fetchkwargs else None
    if provided_bbox and provided_bbox.space_id != image.space_id:
        raise RuntimeError(
            f"Fetching ngbbox error. image.space_id={image.space_id!r} "
            f"!= provided_bbox.space_id={provided_bbox.space_id!r}"
        )

    info_json = get_info(image.url)

    transform_json = get_transform_nm(image.url)

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
    bbox = BoundingBox(
        minpoint=min.tolist(), maxpoint=max.tolist(), space_id=image.space_id
    )

    if not provided_bbox:
        return bbox

    from ...attributes.locations.ops.intersection import bbox_bbox

    return bbox_bbox(bbox, provided_bbox)
