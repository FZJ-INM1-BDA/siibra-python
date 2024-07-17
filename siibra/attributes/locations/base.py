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

from ...attributes import Attribute
from ...commons_new.iterable import assert_ooo


@dataclass
class Location(Attribute):
    schema = "siibra/attr/loc"
    space_id: str = None

    @property
    def space(self):
        if self.space_id is None:
            return None

        from ...factory import iter_collection
        from ...atlases import Space

        return assert_ooo(
            [space for space in iter_collection(Space) if space.ID == self.space_id]
        )


# static methods

# between two locations

# union
# intersection

# instance method

# warp(target_space_id)
# transform(affine)
