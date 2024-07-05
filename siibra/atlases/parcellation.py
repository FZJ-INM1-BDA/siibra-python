from typing import List

from ..atlases import region
from ..commons import logger
from ..commons_new.string import SPEC_TYPE
from ..commons_new.tree import collapse_nodes
from ..commons_new.iterable import assert_ooo
from ..descriptions import Version


class Parcellation(region.Region):
    schema: str = "siibra/atlases/parcellation/v0.1"

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
        from siibra.factory.iterator import iter_collection
        for parc in iter_collection(Parcellation):
            if parc.ID == next_id:
                return parc
        logger.warning(f"Cannot find parcellation with id {next_id}")
        return None

    @property
    def version(self):
        try:
            return self._get(Version).value
        except Exception:
            return None

    @property
    def is_newest_version(self):
        return (self.version is None) or (self._get(Version).next_id is None)
