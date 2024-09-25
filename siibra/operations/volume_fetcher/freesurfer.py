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

from typing import Callable, Dict, List, TYPE_CHECKING
from io import BytesIO
import os

from nibabel import freesurfer, gifti


from .base import PostProcVolProvider, VolumeFormats
from ...operations import DataOp
from ...cache import CACHE
from ...commons.maps import arrs_to_gii

if TYPE_CHECKING:
    from ...attributes.dataproviders.volume import VolumeProvider


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


@VolumeFormats.register_format_read("freesurfer-annot", "mesh")
class FreesurferAnnot(PostProcVolProvider):

    @classmethod
    def on_get_retrieval_ops(cls, volume_provider: "VolumeProvider"):
        base_retrieval_ops = super().on_get_retrieval_ops(volume_provider)
        return [*base_retrieval_ops, ReadGiftiFromBytesFSAAnnot.generate_specs()]


class ReadGiftiFromBytesFSAAnnot(DataOp):
    input: bytes
    output: gifti.GiftiImage
    desc = "Reads bytes into gifti"
    type = "read/freesurfer_annot"

    def run(self, input, **kwargs):
        assert isinstance(input, bytes)
        labels, *_ = freesurfer.read_annot(read_as_bytesio(input))
        return arrs_to_gii(labels)
