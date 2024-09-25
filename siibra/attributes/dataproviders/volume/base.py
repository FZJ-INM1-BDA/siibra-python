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
from typing import TYPE_CHECKING, Tuple, Dict

from ..base import DataProvider
from ....commons.iterable import assert_ooo
from ....commons.logger import logger
from ....operations.volume_fetcher import VolumeFormats
from ....operations.volume_fetcher.base import PostProcVolProvider

if TYPE_CHECKING:
    from ...locations import BoundingBox

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

SIIBRA_MAX_FETCH_SIZE_GIB = getenv("SIIBRA_MAX_FETCH_SIZE_GIB", 0.2)


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

    @property
    def retrieval_ops(self):
        previous_ops = super().retrieval_ops
        return self.volume_postprocess.transform_retrieval_ops(self, previous_ops)

    def append_op(self, op: Dict):
        self.volume_postprocess.on_append_op(self, op)

    def __post_init__(self):
        self.volume_postprocess.on_post_init(self)

    @property
    def volume_postprocess(self):
        try:
            return VolumeFormats.READER_LOOKUP[self.format]
        except KeyError:
            logger.warning(
                f"{self.format} not found in {list(VolumeFormats.READER_LOOKUP.keys())}, default to default reader."
            )
            return PostProcVolProvider

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
