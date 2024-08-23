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

from typing import TYPE_CHECKING
import gzip

from nibabel.gifti import gifti

from ...volume_fetcher.volume_fetcher import FetchKwargs, register_volume_fetcher

if TYPE_CHECKING:
    from ....attributes.dataitems import Mesh


@register_volume_fetcher("gii-mesh", "mesh")
def fetch_gii_mesh(mesh: "Mesh", fetchkwargs: FetchKwargs) -> "gifti.GiftiImage":
    if fetchkwargs["bbox"] is not None:
        raise NotImplementedError
    if fetchkwargs["resolution_mm"] is not None:
        raise NotImplementedError
    if fetchkwargs["color_channel"] is not None:
        raise NotImplementedError

    _bytes = mesh.get_data()
    try:
        return gifti.GiftiImage.from_bytes(gzip.decompress(_bytes))
    except gzip.BadGzipFile:
        return gifti.GiftiImage.from_bytes(_bytes)


@register_volume_fetcher("gii-label", "mesh")
def fetch_gii_label(mesh: "Mesh", fetchkwargs: FetchKwargs) -> "gifti.GiftiImage":
    if fetchkwargs["bbox"] is not None:
        raise NotImplementedError
    if fetchkwargs["resolution_mm"] is not None:
        raise NotImplementedError
    if fetchkwargs["color_channel"] is not None:
        raise NotImplementedError

    _bytes = mesh.get_data()
    try:
        gii = gifti.GiftiImage.from_bytes(gzip.decompress(_bytes))
    except gzip.BadGzipFile:
        gii = gifti.GiftiImage.from_bytes(_bytes)

    return gii
