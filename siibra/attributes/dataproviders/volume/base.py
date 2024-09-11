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

from dataclasses import dataclass
from os import getenv
from typing import TYPE_CHECKING, Tuple, Dict, Type

from ..base import DataProvider
from ....commons.iterable import assert_ooo
from ....commons.logger import logger

if TYPE_CHECKING:
    from ...locations import BoundingBox
    from ....operations.volume_fetcher.base import VolumeRetOp

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

SIIBRA_MAX_FETCH_SIZE_GIB = getenv("SIIBRA_MAX_FETCH_SIZE_GIB", 0.2)
IMAGE_FORMATS = []
MESH_FORMATS = []
FORMAT_LOOKUP = {
    None: IMAGE_FORMATS + MESH_FORMATS,  # TODO maybe buggy
    "mesh": MESH_FORMATS,
    "image": IMAGE_FORMATS,
}
READER_LOOKUP: Dict[str, Type["VolumeRetOp"]] = {}


def register_format_read(format: str, voltype: Literal["mesh", "image"]):
    from ....operations.volume_fetcher.base import VolumeRetOp
    from ....operations.base import DataOp

    def outer(Cls: Type[VolumeRetOp]):
        assert issubclass(
            Cls, VolumeRetOp
        ), f"register_format_read must target a subclass of volume ret op"

        assert issubclass(
            Cls, DataOp
        ), f"register_format_read must target a subclass of data op"
        if format in READER_LOOKUP:
            logger.warning(
                f"{format} already registered by {READER_LOOKUP[format].__name__}, overriden by {Cls.__name__}"
            )
        READER_LOOKUP[format] = Cls
        if voltype == "mesh":
            MESH_FORMATS.append(format)
        if voltype == "image":
            IMAGE_FORMATS.append(format)
        FORMAT_LOOKUP[None] = IMAGE_FORMATS + MESH_FORMATS
        return Cls

    return outer


class Mapping(TypedDict):
    """
    Represents restrictions to apply to an image to get partial information,
    such as labelled mask, a specific slice etc.
    """

    label: int = None
    range: Tuple[float, float]
    subspace: Tuple[slice, ...]
    target: str = None


class VolumeOpsKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.

    Note
    ----
    Not all parameters are avaialble for all formats and volumes.
    """

    bbox: "BoundingBox" = None
    resolution_mm: float = None
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB
    mapping: Dict[str, Mapping] = None


@dataclass
class VolumeProvider(DataProvider):
    schema: str = "siibra/attr/data/volume"
    space_id: str = None
    colormap: str = None  # TODO: remove from config and here
    format: str = None

    def __post_init__(self):
        if self.format not in READER_LOOKUP:
            raise RuntimeError(f"{self.format} cannot be properly parsed as volume")

    @property
    def space(self):
        from .... import find_spaces

        return assert_ooo(
            find_spaces(self.space_id),
            lambda spaces: (
                f"Cannot find any space with the id {self.space_id}"
                if len(spaces) == 0
                else f"Found multiple ({len(spaces)}) spaces with the id {self.space_id}"
            ),
        )

    def assemble_ops(self, **kwargs):
        retrieval_ops, transformation_ops = super().assemble_ops(**kwargs)

        if self.format not in READER_LOOKUP:
            raise RuntimeError(f"{self.format} cannot be properly parsed as volume")
        Cls = READER_LOOKUP[self.format]
        retrieval_ops, transformation_ops = Cls.transform_ops(
            retrieval_ops, transformation_ops, **kwargs
        )
        # post process retrieval
        return retrieval_ops, transformation_ops
