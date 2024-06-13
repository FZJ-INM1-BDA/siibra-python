from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, List, DefaultDict, Set

from ..assignment import iter_attr_col
from ..concepts import AtlasElement
from ..retrieval_new.image_fetcher import FetchKwargs
from ..commons_new.iterable import assert_ooo, get_ooo
from ..commons_new.string import fuzzy_match
from ..atlases import Parcellation
from ..dataitems import Image, VOLUME_FORMATS

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ..locations import BBox
    from ..concepts.attribute_collection import AttributeCollection


@dataclass
class Map(AtlasElement):
    schema: str = "siibra/atlases/parcellation_map/v0.1"
    parcellation_id: str = None
    space_id: str = None
    maptype: str = None
    index_mapping: DefaultDict[str, "AttributeCollection"] = None

    def __post_init__(self):
        essential_specs = {"parcellation_id", "space_id", "maptype", "index_mapping"}
        assert all(
            spec is not None for spec in essential_specs
        ), f"Cannot create a parcellation `Map` without {essential_specs}"
        super().__post_init__()

    @property
    def provides_mesh(self):
        return any(im.provides_mesh for im in self._images)

    @property
    def provides_volume(self):
        return any(im.provides_volume for im in self._images)

    @property
    def parcellation(self) -> "Parcellation":
        return assert_ooo(
            [
                parc
                for parc in iter_attr_col(Parcellation)
                if parc.id == self.parcellation_id
            ]
        )

    @property
    def regions(self) -> List[str]:
        return list(self.index_mapping.keys())

    @property
    def _images(self) -> List["Image"]:
        return [
            im
            for attrcols in self.index_mapping.values()
            for im in attrcols._find(Image)
        ]

    @property
    def image_formats(self) -> Set[str]:
        return {im.format for im in self._images}

    def get_image(self, regionname: str = None, frmt: str = None):

        def filter_format(attr: Image):
            return True if frmt is None else attr.format == frmt

        if regionname is None:
            images = [
                im
                for im in self._images
                if filter_format(im.format)
            ]

        else:
            candidates = [r for r in self.regions if fuzzy_match(regionname, r)]
            selected_region = assert_ooo(candidates)
            images = [
                im
                for im in self.index_mapping[selected_region]._find(Image)
                if filter_format(im.format)
            ]

        return get_ooo(images)

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

    def get_colormap(self, frmt: str = None, regions: List[str] = None) -> List[str]:
        # TODO: should return a matplotlib colormap
        if frmt is None:
            frmt = [f for f in VOLUME_FORMATS if f in self.image_formats][0]
        else:
            assert frmt in self.image_formats

        if regions is None:
            regions = self.regions
        return {
            im.subimage_options["label"]: im.color
            for region in regions
            for im in self.index_mapping[region]._find(Image)
            if im.format == frmt
        }


@dataclass
class SparseMap(Map):

    def __post_init__(self):
        super().__post_init__()
