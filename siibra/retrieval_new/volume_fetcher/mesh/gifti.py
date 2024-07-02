from typing import TYPE_CHECKING
import gzip

from nibabel.gifti import gifti

from ...volume_fetcher.volume_fetcher import FetchKwargs, register_volume_fetcher

if TYPE_CHECKING:
    from ....dataitems import Image


@register_volume_fetcher("gii-mesh", "mesh")
def fetch_gii_mesh(mesh: "Mesh", fetchkwargs: FetchKwargs) -> "gifti.GiftiImage":
    _bytes = mesh.get_data()
    try:
        gii = gifti.GiftiImage.from_bytes(gzip.decompress(_bytes))
    except gzip.BadGzipFile:
        gii = gifti.GiftiImage.from_bytes(_bytes)

    # if fetchkwargs["bbox"] is not None:
    #     # TODO
    #     # neuroglancer/precomputed fetches the bbox from the NeuroglancerScale
    #     gii = extract_voi(gii, fetchkwargs["bbox"])

    # if fetchkwargs["resolution_mm"] is not None:
    #     gii = resample(gii, fetchkwargs["resolution_mm"])

    return gii


@register_volume_fetcher("gii-label", "mesh")
def fetch_gii_label(image: "Image", fetchkwargs: FetchKwargs) -> "gifti.GiftiImage":
    _bytes = image.get_data()
    try:
        gii = gifti.GiftiImage.from_bytes(gzip.decompress(_bytes))
    except gzip.BadGzipFile:
        gii = gifti.GiftiImage.from_bytes(_bytes)

    # if fetchkwargs["bbox"] is not None:
    #     # TODO
    #     # neuroglancer/precomputed fetches the bbox from the NeuroglancerScale
    #     gii = extract_voi(gii, fetchkwargs["bbox"])

    # if fetchkwargs["resolution_mm"] is not None:
    #     gii = resample(gii, fetchkwargs["resolution_mm"])

    return gii
