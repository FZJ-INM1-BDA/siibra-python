from typing import TYPE_CHECKING, Callable
from io import BytesIO
import os

from nibabel import freesurfer, gifti

from ..volume_fetcher import FetchKwargs, register_volume_fetcher
from ....cache import CACHE
from ....commons_new.maps import arrs_to_gii

if TYPE_CHECKING:
    from ....dataitems import Mesh


def read_as_bytesio(function: Callable, suffix: str, bytesio: BytesIO):
    """
    Helper method to provide BytesIO to methods that only takes file path and
    cannot handle BytesIO normally (e.g., `nibabel.freesurfer.read_annot()`).

    Writes the bytes to a temporary file on cache and reads with the
    original function.

    Parameters
    ----------
    function : Callable
    suffix : str
        Must match the suffix expected by the function provided.
    bytesio : BytesIO

    Returns
    -------
    Return type of the provided function.
    """
    tempfile = CACHE.build_filename(f"temp_{suffix}") + suffix
    with open(tempfile, "wb") as bf:
        bf.write(bytesio.getbuffer())
    result = function(tempfile)
    os.remove(tempfile)
    return result


@register_volume_fetcher("freesurfer-annot", "mesh")
def fetch_freesurfer_annot(mesh: "Mesh", fetchkwargs: FetchKwargs) -> gifti.GiftiImage:
    if fetchkwargs["bbox"] is not None:
        raise NotImplementedError
    if fetchkwargs["resolution_mm"] is not None:
        raise NotImplementedError
    if fetchkwargs["color_channel"] is not None:
        raise NotImplementedError
    labels, *_ = freesurfer.read_annot(read_as_bytesio(mesh.get_data()))
    return arrs_to_gii(labels)
