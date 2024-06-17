from typing import TYPE_CHECKING, TypedDict, Callable, Dict
from functools import wraps

from ...commons import SIIBRA_MAX_FETCH_SIZE_GIB


if TYPE_CHECKING:
    from ...locations import BBox
    from ...dataitems import Image
    from nibabel import Nifti1Image


class FetchKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.
    """

    bbox: "BBox" = None
    resolution_mm: float = None
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB
    color_channel: int = None


_REGISTRY: Dict[str, Callable[[Image, FetchKwargs], Nifti1Image]] = {}


def register_image_fetcher(format: str):

    def outer(fn: Callable[[Image, FetchKwargs], Nifti1Image]):

        @wraps(fn)
        def inner(image: Image, fetchkwargs: FetchKwargs):
            assert image.format == format, f"Expected {format}, but got {image.format}"
            return fn(image, fetchkwargs)

        _REGISTRY[format] = inner
        return inner

    return outer


def get_image_fetcher(format: str):
    assert format in _REGISTRY, f"{format} not found in registry"
    return _REGISTRY[format]


@register_image_fetcher("nii")
def handle_nifti_fetch(image: Image, fetchkwargs: FetchKwargs):
    image.get_data()


@register_image_fetcher("neuroglancer/precomputed")
def handle_neuroglancer_fetch(image: Image, fetchkwargs: FetchKwargs):
    fetchkwargs["bbox"]
    fetchkwargs["resolution_mm"]
    pass
