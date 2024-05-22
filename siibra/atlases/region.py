from dataclasses import dataclass
from typing import Tuple

import anytree

from ..concepts import atlas_elements


@dataclass
class Region(
    anytree.NodeMixin,
    atlas_elements.AtlasElement
):
    schema: str = "siibra/attrCln/atlasEl/region"
    parent: "Region" = None
    children: Tuple["Region"] = None

    @property
    def parcellation(self):
        return self.root

    def tree2str(self):
        """Render region-tree as a string"""
        return "\n".join(
            "%s%s" % (pre, node.name)
            for pre, _, node
            in anytree.RenderTree(self, style=anytree.render.ContRoundStyle)
        )

    def render_tree(self):
        """Prints the tree representation of the region"""
        print(self.tree2str())
