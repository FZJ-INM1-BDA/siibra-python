from typing import TYPE_CHECKING, TypedDict, Callable, Dict, Union, Literal
from functools import wraps

from ...commons import SIIBRA_MAX_FETCH_SIZE_GIB


if TYPE_CHECKING:
    from ...locations import BBox
    from ...dataitems import Image, Mesh
    from nibabel import Nifti1Image, GiftiImage


class FetchKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.
    """

    bbox: "BBox" = None
    resolution_mm: float = None
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB
    color_channel: int = None


# TODO: Reconsider this approach. Only really exists for fsaverage templates
VARIANT_KEY = "x-siibra/volume-variant"
FRAGMENT_KEY = "x-siibra/volume-fragment"

FETCHER_REGISTRY: Dict[
    str, Callable[["Image", FetchKwargs], Union["Nifti1Image", "GiftiImage"]]
] = {}
BBOX_GETTER_REGISTRY: Dict[str, Callable[["Image", FetchKwargs], "BBox"]] = {}
IMAGE_FORMATS = []
MESH_FORMATS = []


def register_volume_fetcher(format: str, volume_type: Literal["image", "mesh"]):

    def outer(
        fn: Callable[
            [Union["Image", "Mesh"], FetchKwargs], Union["Nifti1Image", "GiftiImage"]
        ]
    ):

        @wraps(fn)
        def inner(volume: Union["Image", "Mesh"], fetchkwargs: FetchKwargs):
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

    def outer(fn: Callable[["Image", FetchKwargs], "BBox"]):

        @wraps(fn)
        def inner(image: "Image", fetchkwargs: FetchKwargs):
            assert image.format == format, f"Expected {format}, but got {image.format}"
            return fn(image, fetchkwargs)

        BBOX_GETTER_REGISTRY[format] = inner

        return inner

    return outer


def get_bbox_getter(format: str):
    assert format in BBOX_GETTER_REGISTRY, f"{format} not found in registry"
    return BBOX_GETTER_REGISTRY[format]
