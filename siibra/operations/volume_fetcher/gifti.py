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

import gzip
from typing import TYPE_CHECKING, List, Dict
from nibabel.gifti import gifti

from .base import PostProcVolProvider, VolumeFormats
from ...operations import DataOp
from ...commons.maps import _merge_giftis

if TYPE_CHECKING:
    from ...attributes.dataproviders.volume import VolumeProvider


@VolumeFormats.register_format_read("gii-mesh", "mesh")
@VolumeFormats.register_format_read("gii-label", "mesh")
class FreesurferAnnot(PostProcVolProvider):

    @classmethod
    def on_get_retrieval_ops(cls, volume_provider: "VolumeProvider"):
        base_retrieval_ops = super().on_get_retrieval_ops(volume_provider)
        return [*base_retrieval_ops, ReadGiftiFromBytesGii.generate_specs()]


class ReadGiftiFromBytesGii(DataOp):
    input: bytes
    output: gifti.GiftiImage
    desc = "Reads bytes into gifti"
    type = "volume/gifti/read"

    def run(self, input, **kwargs):
        assert isinstance(input, bytes)
        try:
            return gifti.GiftiImage.from_bytes(gzip.decompress(input))
        except gzip.BadGzipFile:
            return gifti.GiftiImage.from_bytes(input)


class MergeGifti(DataOp):
    input: List[gifti.GiftiImage]
    output: gifti.GiftiImage
    desc = "Merge multiple giftis into one"
    type = "volume/gifti/merge"

    def run(self, input, **kwargs):
        return _merge_giftis(input)