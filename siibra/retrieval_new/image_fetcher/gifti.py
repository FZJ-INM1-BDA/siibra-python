from typing import TYPE_CHECKING
import gzip

from nibabel import GiftiImage

from .image_fetcher import register_image_fetcher, FetchKwargs

if TYPE_CHECKING:
    from ...dataitems import Image
    from ...locations import BBox


def extract_voi(gifti: GiftiImage, voi: "BBox"):
    raise NotImplementedError


def resample(gifti: GiftiImage, resolution_mm: float = None, affine=None):
    raise NotImplementedError


class GiftiLabelFetcher:

    def __init__(self):
        pass

    @register_image_fetcher("gii-mesh")
    @register_image_fetcher("gii-label")
    def fetch(image: "Image", fetchkwargs: FetchKwargs) -> "GiftiImage":
        _bytes = image.get_data()
        if _bytes.startswith(b'\x1f\x8b'):
            _bytes = gzip.decompress(_bytes)
        gii = GiftiImage.from_bytes(_bytes)

        if fetchkwargs["bbox"] is not None:
            # TODO
            # neuroglancer/precomputed fetches the bbox from the NeuroglancerScale
            gii = extract_voi(gii, fetchkwargs["bbox"])

        if fetchkwargs["resolution_mm"] is not None:
            gii = resample(gii, fetchkwargs["resolution_mm"])

        return gii
