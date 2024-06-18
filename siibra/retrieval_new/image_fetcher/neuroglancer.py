from typing import TYPE_CHECKING

from .image_fetcher import register_image_fetcher

if TYPE_CHECKING:
    from nibabel import Nifti1Image, GiftiImage


class NeuroglancerFetcher:
    @register_image_fetcher("neuroglancer/precomputed")
    def fetch(self) -> "Nifti1Image":
        pass


class NeuroglancerMeshFetcher:

    def __init__(self):
        pass

    @register_image_fetcher("neuroglancer/precompmesh")
    def fetch(self) -> "GiftiImage":
        pass
