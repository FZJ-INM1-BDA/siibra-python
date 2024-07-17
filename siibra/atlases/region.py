from typing import Iterable, Union, TYPE_CHECKING, List

import anytree

from ..concepts import atlas_elements
from ..attributes.descriptions import Name
from ..commons_new.string import get_spec, SPEC_TYPE
from ..commons_new.iterable import assert_ooo
from ..commons_new.maps import spatial_props
from ..commons_new.logger import logger

if TYPE_CHECKING:
    from .space import Space
    from ..attributes.locations import PointCloud
    from . import Parcellation


def filter_newest(regions: List["Region"]) -> List["Region"]:
    _parcellations = {r.parcellation for r in regions}
    return [r for r in regions if r.parcellation.next_version not in _parcellations]


class Region(atlas_elements.AtlasElement, anytree.NodeMixin):
    schema: str = "siibra/atlases/region/v0.1"

    def __hash__(self):
        return hash(self.parcellation.ID + self.name)

    def __init__(self, attributes, children):
        super().__init__(self, attributes=attributes)
        anytree.NodeMixin.__init__(self)
        self.children = children

    def __repr__(self) -> str:
        if self == self.root:
            return atlas_elements.AtlasElement.__repr__(self)
        return f"{self.__class__.__name__}({self.name!r} in {self.parcellation.name!r})"

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
        from ..locations import PointCloud

        spatialprops = self._get_spatialprops(space=space, maptype="labelled")
        return PointCloud([sp["centroid"] for sp in spatialprops], space_id=space.ID)

    def get_components(self, space: Union[str, "Space", None] = None):
        spatialprops = self._get_spatialprops(space=space, maptype="labelled")
        return [sp["volume"] for sp in spatialprops]

    def _get_spatialprops(
        self,
        space: Union[str, "Space", None] = None,
        maptype: str = "labelled",
        threshold: float = 0.0,
    ):
        mask = self.find_regional_maps(space=space, maptype=maptype)
        return spatial_props(
            mask, space_id=space.ID, maptype=maptype, threshold_statistical=threshold
        )

    def _finditer_regional_maps(
        self, space: Union[str, "Space", None] = None, maptype: str = "labelled"
    ):
        from .space import Space
        from .parcellationmap import Map, VALID_MAPTYPES
        from ..assignment import string_search
        from ..factory import iter_collection

        assert (
            maptype in VALID_MAPTYPES
        ), f"maptypes can be in {VALID_MAPTYPES}, but you provided {maptype}"

        if isinstance(space, str):
            space = assert_ooo(list(string_search(space, Space)))

        assert space is None or isinstance(
            space, Space
        ), f"space must be str, Space or None. You provided {space}"

        for mp in iter_collection(Map):
            if maptype != mp.maptype:
                continue
            if space and space.ID != mp.space_id:
                continue
            if self.parcellation.ID != mp.parcellation_id:
                continue
            mapped_regions = [r for r in self if r.name in mp.regions]
            if len(mapped_regions) == 0:
                continue
            yield mp.get_filtered_map(mapped_regions)

    def find_regional_maps(
        self, space: Union[str, "Space", None] = None, maptype: str = "labelled"
    ):
        return list(self._finditer_regional_maps(space, maptype))

    def fetch_regional_map(
        self,
        space: Union[str, "Space", None] = None,
        maptype: str = "labelled",
        threshold: float = 0.0,
        via_space: Union[str, "Space", None] = None,
        frmt: str = None,
    ):
        if via_space is not None:
            raise NotImplementedError
        if threshold != 0.0:
            raise NotImplementedError
        maps = self.find_regional_maps(space=space, maptype=maptype)
        try:
            selectedmap = assert_ooo(maps)
        except AssertionError:
            if maps:
                logger.warning(f"Found {len(maps)} maps matching the specs. Selecting the first.")
                selectedmap = maps[0]
            else:
                raise ValueError("Found no maps matching the specs for this region.")

        return selectedmap.fetch(frmt=frmt)
