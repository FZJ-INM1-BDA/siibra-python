from dataclasses import dataclass
from typing import Iterable, Union, TYPE_CHECKING

import anytree

from ..concepts import atlas_elements
from ..descriptions import Name
from ..commons_new.string import get_spec, SPEC_TYPE
from ..commons_new.iterable import assert_ooo

if TYPE_CHECKING:
    from nibabel import Nifti1Image
    from .space import Space
    from ..locations import PointCloud
    from . import Parcellation


@dataclass
class Region(atlas_elements.AtlasElement, anytree.NodeMixin):
    schema: str = "siibra/atlases/region/v0.1"

    def __init__(self, attributes, children):
        super().__init__(self, attributes=attributes)
        anytree.NodeMixin.__init__(self)
        self.children = children

    def __eq__(self, other: "Region") -> bool:
        # Otherwise, == comparison result in max stack
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Region<{self.name!r} in {self.parcellation.name!r}>"

    @property
    def parcellation(self) -> "Parcellation":
        return self.root

    def tree2str(self):
        """Render region-tree as a string"""
        return "\n".join(
            "%s%s" % (pre, node.name)
            for pre, _, node in anytree.RenderTree(
                self, style=anytree.render.ContRoundStyle
            )
        )

    def __iter__(self):
        return anytree.PreOrderIter(self)

    def render_tree(self):
        """Prints the tree representation of the region"""
        print(self.tree2str())

    def find(self, regionspec: SPEC_TYPE):
        children: Iterable[Region] = (region for region in anytree.PreOrderIter(self))
        match_fn = get_spec(regionspec)
        return [
            child
            for child in children
            if any(match_fn(name.value) for name in child._finditer(Name))
        ]

    def get_centroids(self, space: Union[str, "Space", None] = None) -> "PointCloud":
        raise NotImplementedError

    def _finditer_regional_maps(
        self, space: Union[str, "Space", None] = None, maptype: str = "LABELLED"
    ):
        from .space import Space
        from ..assignment import iter_attr_col, string_search
        from .parcellationmap import Map, VALID_MAPTYPES

        assert maptype in VALID_MAPTYPES, f"maptypes can be in {VALID_MAPTYPES}, but you provided {maptype}"

        if isinstance(space, str):
            space = assert_ooo(string_search(space, Space))

        assert space is None or isinstance(
            space, Space
        ), f"space must be str, Space or None. You provided {space}"

        regions_of_interest = [self, *self.children]

        for mp in iter_attr_col(Map):
            if maptype != mp.maptype:
                continue
            if space and space.id != mp.space_id:
                continue
            if self.parcellation.id != mp.parcellation_id:
                continue
            yield mp.filter_regions(regions_of_interest)

    def find_regional_maps(self, space: Union[str, "Space", None] = None, maptype: str = "LABELLED"):
        return list(self._finditer_regional_maps(space, maptype))

    def fetch_regional_map(
        self,
        space: Union[str, "Space", None] = None,
        maptype: str = "LABELLED",
        threshold: float = 0.0,
        via_space: Union[str, "Space", None] = None
    ) -> "Nifti1Image":
        pass
