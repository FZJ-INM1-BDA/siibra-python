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
from typing import TYPE_CHECKING

from ...attributes import Attribute
from ...commons.iterable import assert_ooo

if TYPE_CHECKING:
    from numpy import ndarray


@dataclass
class Location(Attribute):
    schema = None
    space_id: str = None

    @property
    def space(self):
        if self.space_id is None:
            return None

        from ...factory import iter_preconfigured
        from ...atlases import Space

        return assert_ooo(
            [space for space in iter_preconfigured(Space) if space.ID == self.space_id]
        )

    def union(self, other: "Location"):
        from .ops.union import union as _union

        return _union(self, other)

    def intersect(self, other: "Location"):
        from .ops.intersection import intersect as _intersect

        return _intersect(self, other)

    def warp(self, space_id: str):
        from .ops.warp import warp as _warp

        return _warp(self, space_id)

    def transform(self, affine: "ndarray", space_id: str = None):
        from .ops.transform import transform as _transform

        return _transform(self, affine, space_id)
