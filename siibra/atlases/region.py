from dataclasses import dataclass
from typing import Iterable

import anytree

from ..concepts import atlas_elements
from ..descriptions import Name
from ..commons_new.string import get_spec, SPEC_TYPE


@dataclass
class Region(atlas_elements.AtlasElement, anytree.NodeMixin):
    schema: str = "siibra/atlases/region/v0.1"

    def __init__(self, attributes, children):
        super().__init__(self, attributes=attributes)
        anytree.NodeMixin.__init__(self)
        self.children = children

    def __eq__(self, other: "Region") -> bool:
        # Otherwise, == comparisonn result in max stack
        return self.ID == other.ID

    def __hash__(self):
        return hash(self.ID)

    @property
    def parcellation(self):
        return self.root

    def tree2str(self):
        """Render region-tree as a string"""
        return "\n".join(
            "%s%s" % (pre, node.name)
            for pre, _, node in anytree.RenderTree(
                self, style=anytree.render.ContRoundStyle
            )
        )

    def render_tree(self):
        """Prints the tree representation of the region"""
        print(self.tree2str())

    def find(self, regionspec: SPEC_TYPE):
        children: Iterable[Region] = (region for region in anytree.PreOrderIter(self))
        match_fn = get_spec(regionspec)
        return [
            child
            for child in children
            if any(match_fn(name.value) for name in child.getiter(Name))
        ]
