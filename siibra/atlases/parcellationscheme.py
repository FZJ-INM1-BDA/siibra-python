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

from typing import List, Union, TYPE_CHECKING

from . import region
from ..commons.logger import logger
from ..commons.string import SPEC_TYPE
from ..commons.tree import collapse_nodes
from ..commons.iterable import assert_ooo
from ..attributes.descriptions import Version

if TYPE_CHECKING:
    from .space import Space


class ParcellationScheme(region.Region):
    schema: str = "siibra/atlases/parcellationscheme/v0.1"

    def __lt__(self, other) -> bool:
        assert isinstance(other, type(self)), TypeError(
            f"'>' not supported between instances of '{type(self)}' and '{type(other)}'"
        )
        try:
            assert (self.version is not None) and (other.version is not None)
        except AssertionError:
            raise TypeError("Cannot compare Parcellations with no version information")
        return self.version.prev_id == other.version.next_id

    def __eq__(self, other: "ParcellationScheme") -> bool:
        return self.ID == other.ID

    def __hash__(self):
        return hash(self.ID)

    def get_region(self, regionspec: SPEC_TYPE) -> region.Region:
        """
        Returns a single collapsed region, based on regionspec

        n.b. collapsing the region tree means, recursively, if all children of a node is selected in get_region,
        the parnet is selected instead."""
        regions = self.find(regionspec)
        exact_match = [region for region in regions if region.name == regionspec]
        if len(exact_match) == 1:
            return exact_match[0]
        collapsed_regions: List[region.Region] = collapse_nodes(regions)
        return assert_ooo(collapsed_regions)

    @property
    def next_version(self):
        if not self.version:
            return None
        next_id = self._get(Version).next_id
        if not next_id:
            return None
        from siibra.factory import iter_preconfigured

        for parc in iter_preconfigured(ParcellationScheme):
            if parc.ID == next_id:
                return parc
        logger.warning(f"Cannot find parcellation with id {next_id}")
        return None

    @property
    def version(self):
        try:
            return self._get(Version)
        except Exception:
            return None

    @property
    def is_newest_version(self):
        return (self.version is None) or (self._get(Version).next_id is None)

    def find_maps(
        self,
        space: Union["Space", str, None] = None,
        maptype: Union[str, None] = None,
        name: str = "",
    ):
        from .. import find_maps

        return find_maps(self.ID, space, maptype=maptype, name=name)

    def get_map(
        self,
        space: Union["Space", str, None] = None,
        maptype: Union[None, str] = None,
        name: str = "",
    ):
        searched_maps = self.find_maps(space, maptype, name)
        return assert_ooo(searched_maps)
