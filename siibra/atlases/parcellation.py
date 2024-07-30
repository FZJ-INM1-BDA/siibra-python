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

from typing import List

from ..atlases import region
from ..commons_new.logger import logger
from ..commons_new.string import SPEC_TYPE
from ..commons_new.tree import collapse_nodes
from ..commons_new.iterable import assert_ooo
from ..attributes.descriptions import Version


class Parcellation(region.Region):
    schema: str = "siibra/atlases/parcellation/v0.1"

    def __lt__(self, other) -> bool:
        assert isinstance(other, type(self)), TypeError(f"'>' not supported between instances of '{type(self)}' and '{type(other)}'")
        try:
            assert (self.version is not None) and (other.version is not None)
        except AssertionError:
            raise TypeError("Cannot compare Parcellations with no version information")
        return self.version.prev_id == other.version.next_id

    def __eq__(self, other: "Parcellation") -> bool:
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
        from siibra.factory.iterator import iter_preconfigured_ac
        for parc in iter_preconfigured_ac(Parcellation):
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

    def find_maps(self, space_id: str = None, maptype: str = "labelled", extra_spec: str = ""):
        # TODO: reconsider the placement of this method and reference them at the package level
        from ..factory import iter_preconfigured_ac
        from . import Map

        return_result = []
        for _map in iter_preconfigured_ac(Map):
            if _map.maptype != maptype:
                continue
            if _map.parcellation != self:
                continue
            if _map.space_id != space_id:
                continue
            if extra_spec not in _map.name:
                continue
            return_result.append(_map)

        return return_result

    def get_map(self, space_id: str = None, maptype: str = "labelled", extra_spec: str = ""):
        # TODO: reconsider the placement of this method and reference them at the package level
        searched_maps = self.find_maps(space_id, maptype, extra_spec)
        return assert_ooo(searched_maps)
