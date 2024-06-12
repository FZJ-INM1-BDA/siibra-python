from dataclasses import dataclass, asdict
from typing import Tuple, TYPE_CHECKING, Set

from ..assignment import iter_attr_col
from ..concepts import AtlasElement
from ..retrieval_new.image_fetcher import FetchKwargs
from ..descriptions import RGBColor
from ..commons_new.iterable import assert_ooo
from ..commons_new.string import fuzzy_match
from ..atlases import Parcellation
from ..dataitems import Image

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ..locations import BBox


@dataclass
class Map(AtlasElement):
    schema: str = "siibra/atlases/parcellation_map/v0.1"
    parcellation_id: str = None
    space_id: str = None
    maptype: str = None

    def __post_init__(self):
        essential_specs = {"parcellation_id", "space_id", "maptype"}
        assert all(
            spec is not None for spec in essential_specs
        ), f"Cannot create a parcellation `Map` without {essential_specs}"
        super().__post_init__()

        self._images = self._find(Image)

    @property
    def parcellation(self) -> "Parcellation":
        return assert_ooo([parc
                           for parc in iter_attr_col(Parcellation)
                           if parc.id == self.parcellation_id])

    @property
    def regions(self) -> Tuple[str]:
        return ((im.extra["x-regionname"] for im in self._images),)

    def _get_image_keys(self, regionname: str) -> Set[str]:
        try:
            matched_regions = {
                r.name
                for r in (self.parcellation.find(regionname))
                if r.name in self.regions
            }
            matched_region = assert_ooo(matched_regions)
        except AssertionError:
            raise RuntimeError(
                f"'{regionname}' matches several regions in this map: {matched_regions}"
            )  # TODO: create a raise type
        return {index.get("key") for index in self._indices[matched_region]}

    def get_image(self, regionname: str = None, frmt: str = None):
        from ..dataitems import Image

        def filter_fn(im: Image):
            if regionname is None:
                if frmt is None:
                    return True
                return frmt == im.format

            if fuzzy_match(regionname, im.extra["x-regionname"]):
                if frmt:
                    return frmt == im.format
                return True

            return False

        images = [im for im in self._images if filter_fn(im)]
        return assert_ooo(images)

    def fetch(
        self,
        regionname: str = None,
        frmt: str = None,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_bytes: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
        label: int = None,
    ):
        image = self.get_image(regionname=regionname, frmt=frmt)
        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_bytes,
            label=label,
        )
        return image.fetch(**asdict(fetch_kwargs))

    def get_colormap(self):
        # TODO
        # should return a matplotlib colormap
        # also, the rgb value should be stored in the map
        clrs = {r: assert_ooo(self._get(RGBColor)) for r in self.regions}
        return clrs


@dataclass
class SparseMap(Map):
    pass
