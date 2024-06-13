from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING, List, DefaultDict, Set

from ..assignment import iter_attr_col
from ..concepts import AtlasElement
from ..retrieval_new.image_fetcher import FetchKwargs
from ..commons_new.iterable import assert_ooo
from ..commons_new.string import fuzzy_match
from ..commons_new.nifti_operations import resample_to_template_and_merge
from ..atlases import Parcellation, Space
from ..dataitems import Image, IMAGE_FORMATS

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
        return any(im.provides_mesh for im in self._region_images)

    @property
    def provides_volume(self):
        return any(im.provides_volume for im in self._region_images)

    @property
    def parcellation(self) -> "Parcellation":
        return assert_ooo([
            parc
            for parc in iter_attr_col(Parcellation)
            if parc.id == self.parcellation_id
        ])

    @property
    def space(self) -> "Space":
        return assert_ooo([
            sp
            for sp in iter_attr_col(Space)
            if sp.id == self.space_id
        ])

    @property
    def regions(self) -> Set[str]:
        return set(self.index_mapping.keys())

    @property
    def _region_images(self) -> List["Image"]:
        return [
            im
            for attrcols in self.index_mapping.values()
            for im in attrcols._find(Image)
        ]

    @property
    def image_formats(self) -> Set[str]:
        return {im.format for im in self._region_images}

    def _find_images(self, regionname: str = None, frmt: str = None) -> List["Image"]:

        def filter_format(attr: Image):
            return True if frmt is None else attr.format == frmt

        if regionname is None:
            return [img for img in self._find(Image) if filter_format(img)]

        if regionname in self.regions:
            selected_region = regionname
        else:
            candidates = [r for r in self.regions if fuzzy_match(regionname, r)]
            selected_region = assert_ooo(candidates)

        return [
            img
            for img in self.index_mapping[selected_region]._find(Image)
            if filter_format(img)
        ]

    def fetch(
        self,
        regionname: str = None,
        frmt: str = None,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_bytes: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None
    ):
        if frmt is None:
            frmt = [f for f in IMAGE_FORMATS if f in self.image_formats][0]
        else:
            assert frmt in self.image_formats, f"Requested format '{frmt}' is not available for this map: {self.image_formats=}."

        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_bytes
        )

        images = self._find_images(regionname=regionname, frmt=frmt)
        if len(images) == 1:
            return images[0].fetch(**asdict(fetch_kwargs))
        elif len(images) > 1:
            template = self.space.get_template(fetch_kwargs=fetch_kwargs)
            # TODO: fix relabelling
            # labels = [im.subimage_options["label"] for im in self._region_images]
            # if set(labels) == {1}:
            #     labels = list(range(1, len(labels) + 1))
            return resample_to_template_and_merge(
                images, template, labels=[]
            )

    def get_colormap(self, frmt: str = None, regions: List[str] = None) -> List[str]:
        from matplotlib.colors import ListedColormap
        import numpy as np

        def convert_hex_to_tuple(clr: str):
            return tuple(int(clr[p:p + 2], 16) for p in [1, 3, 5])

        if frmt is None:
            frmt = [f for f in IMAGE_FORMATS if f in self.image_formats][0]
        else:
            assert frmt in self.image_formats, f"Requested format '{frmt}' is not available for this map: {self.image_formats=}."

        if regions is None:
            regions = self.regions

        colors = {
            im.subimage_options["label"]: convert_hex_to_tuple(im.color)
            for region in regions
            for im in self._find_images(region, frmt=frmt)
        }
        pallette = np.array(
            [
                list(colors[i]) + [1] if i in colors else [0, 0, 0, 0]
                for i in range(max(colors.keys()) + 1)
            ]
        ) / [255, 255, 255, 1]
        return ListedColormap(pallette)


@dataclass
class SparseMap(Map):

    def __post_init__(self):
        super().__post_init__()
