from dataclasses import dataclass
from typing import List, Iterable

import anytree

from ..concepts import atlas_elements, QueryParam
from ..descriptions import ID, Name
from ..assignment import filter_collections


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

    def find(self, regionspec: str):
        children: Iterable[Region] = (region for region in anytree.PreOrderIter(self))
        id_attr = ID(value=regionspec)
        name_attr = Name(value=regionspec)
        query = QueryParam(attributes=[id_attr, name_attr])
        yield from filter_collections(query, children)
