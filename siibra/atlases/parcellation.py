from dataclasses import dataclass
from typing import List

from ..atlases import region
from ..commons_new.string import SPEC_TYPE
from ..commons_new.tree import collapse_nodes
from ..commons_new.iterable import assert_ooo


@dataclass(init=False)
class Parcellation(region.Region):
    schema: str = "siibra/atlases/parcellation/v0.1"

    def get_region(self, regionspec: SPEC_TYPE):
        """
        Returns a single collapsed region, based on regionspec

        n.b. collapsing the region tree means, recursively, if all children of a node is selected in get_region,
        the parnet is selected instead."""
        regions = self.find(regionspec)
        collapsed_regions: List[region.Region] = collapse_nodes(regions)
        return assert_ooo(collapsed_regions)
