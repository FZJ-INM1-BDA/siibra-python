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

from typing import TYPE_CHECKING, Callable, Dict, Union, Tuple
from os import getenv
from functools import wraps

try:
    from typing import TypedDict, Literal
except ImportError:
    # support python 3.7
    from typing_extensions import TypedDict, Literal

SIIBRA_MAX_FETCH_SIZE_GIB = getenv("SIIBRA_MAX_FETCH_SIZE_GIB", 0.2)

if TYPE_CHECKING:
    from ...attributes.locations import BoundingBox
    from ...attributes.dataproviders import ImageProvider, MeshProvider
    from nibabel import Nifti1Image, GiftiImage


class Mapping(TypedDict):
    """
    Represents restrictions to apply to an image to get partial information,
    such as labelled mask, a specific slice etc.
    """

    label: int = None
    range: Tuple[float, float]
    subspace: Tuple[slice, ...]
    t: int


class FetchKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.

    Note
    ----
    Not all parameters are avaialble for all formats and volumes.
    """

    bbox: "BoundingBox" = None
    resolution_mm: float = None
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB
    mapping: Dict[str, Mapping] = None


# TODO: Reconsider this approach. Only really exists for fsaverage templates
FRAGMENT_KEY = "x-siibra/volume-fragment"

FETCHER_REGISTRY: Dict[
    str, Callable[["ImageProvider", FetchKwargs], Union["Nifti1Image", "GiftiImage"]]
] = {}
BBOX_GETTER_REGISTRY: Dict[str, Callable[["ImageProvider", FetchKwargs], "BoundingBox"]] = {}
IMAGE_FORMATS = []
MESH_FORMATS = []


def register_volume_fetcher(format: str, volume_type: Literal["image", "mesh"]):
    def outer(
        fn: Callable[
            [Union["ImageProvider", "MeshProvider"], FetchKwargs], Union["Nifti1Image", "GiftiImage"]
        ]
    ):
        @wraps(fn)
        def inner(volume: Union["ImageProvider", "MeshProvider"], fetchkwargs: FetchKwargs):
            assert (
                volume.format == format
            ), f"Expected {format}, but got {volume.format}"
            return fn(volume, fetchkwargs)

        FETCHER_REGISTRY[format] = inner
        if volume_type == "mesh":
            MESH_FORMATS.append(format)
        elif volume_type == "image":
            IMAGE_FORMATS.append(format)
        else:
            raise ValueError(f"'{volume_type}' is not a valid image type.")

        return inner

    return outer


def get_volume_fetcher(format: str):
    assert format in FETCHER_REGISTRY, f"{format} not found in registry"
    return FETCHER_REGISTRY[format]


def register_bbox_getter(format: str):
    def outer(fn: Callable[["ImageProvider", Union[FetchKwargs, None]], "BoundingBox"]):
        assert (
            format not in BBOX_GETTER_REGISTRY
        ), f"format={format!r} already registered bbox getter"
        BBOX_GETTER_REGISTRY[format] = fn
        return fn

    return outer


def get_bbox_getter(format: str):
    assert format in BBOX_GETTER_REGISTRY, f"{format} not found in registry"
    return BBOX_GETTER_REGISTRY[format]
