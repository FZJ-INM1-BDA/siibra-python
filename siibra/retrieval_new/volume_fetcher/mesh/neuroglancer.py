from typing import TYPE_CHECKING

from ...volume_fetcher.volume_fetcher import register_volume_fetcher

if TYPE_CHECKING:
    from nibabel import GiftiImage


@register_volume_fetcher("neuroglancer/precompmesh", "mesh")
def fetch_neuroglancer_mesh(mesh) -> "GiftiImage":
    pass
