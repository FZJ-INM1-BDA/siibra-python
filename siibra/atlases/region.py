from dataclasses import dataclass
from typing import List

import anytree

from ..concepts import atlas_elements


@dataclass
class Region(
    atlas_elements.AtlasElement,
    anytree.NodeMixin
):
    schema: str = "siibra/atlases/region/v0.1"

    def __init__(self, attributes, children):
        super().__init__(self, attributes=attributes)
        anytree.NodeMixin.__init__(self)
        self.children = children

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
