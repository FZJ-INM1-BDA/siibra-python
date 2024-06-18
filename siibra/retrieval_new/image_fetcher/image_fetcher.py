from typing import TYPE_CHECKING, TypedDict, Callable, Dict, Union, Literal
from functools import wraps

from ...commons import SIIBRA_MAX_FETCH_SIZE_GIB


if TYPE_CHECKING:
    from ...locations import BBox
    from ...dataitems import Image
    from nibabel import Nifti1Image, GiftiImage


class FetchKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.
    """

    bbox: "BBox" = None
    resolution_mm: float = None
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB
    color_channel: int = None


_REGISTRY: Dict[
    str, Callable[["Image", FetchKwargs], Union["Nifti1Image", "GiftiImage"]]
] = {}
VOLUME_FORMATS = []
MESH_FORMATS = []


# TODO: consider predecorating with fn_call_cache
def register_image_fetcher(format: str, image_type: Literal["volume", "mesh"]):

    def outer(fn: Callable[["Image", FetchKwargs], Union["Nifti1Image", "GiftiImage"]]):

        @wraps(fn)
        def inner(image: "Image", fetchkwargs: FetchKwargs):
            assert image.format == format, f"Expected {format}, but got {image.format}"
            return fn(image, fetchkwargs)

        _REGISTRY[format] = inner
        if image_type == "mesh":
            MESH_FORMATS.append(format)
        elif image_type == "volume":
            VOLUME_FORMATS.append(format)
        else:
            raise ValueError(f"'{image_type}' is not a valid image type.")

        return inner

    return outer


def get_image_fetcher(format: str):
    assert format in _REGISTRY, f"{format} not found in registry"
    return _REGISTRY[format]
