from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Dict, Set, Union

import numpy as np

from ..assignment import iter_attr_col
from ..concepts import AtlasElement
from ..retrieval_new.image_fetcher import FetchKwargs
from ..commons_new.iterable import assert_ooo
from ..commons_new.maps import resample_to_template_and_merge
from ..atlases import Parcellation, Space, Region
from ..dataitems import Image, IMAGE_FORMATS
from ..descriptions import Name, ID as _ID, SpeciesSpec

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ..locations import BBox
    from ..concepts.attribute_collection import AttributeCollection


VALID_MAPTYPES = ("STATISTICAL", "LABELLED")


@dataclass
class Map(AtlasElement):
    schema: str = "siibra/atlases/parcellation_map/v0.1"
    parcellation_id: str = None
    space_id: str = None
    maptype: str = None
    _index_mapping: Dict[str, "AttributeCollection"] = field(default_factory=dict)

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
        return assert_ooo(
            [
                parc
                for parc in iter_attr_col(Parcellation)
                if parc.ID == self.parcellation_id
            ]
        )

    @property
    def space(self) -> "Space":
        return assert_ooo([sp for sp in iter_attr_col(Space) if sp.ID == self.space_id])

    @property
    def regions(self) -> List[str]:
        return list(self._index_mapping.keys())

    @property
    def _region_images(self) -> List["Image"]:
        return [
            im
            for attrcols in self._index_mapping.values()
            for im in attrcols._find(Image)
        ]

    @property
    def image_formats(self) -> Set[str]:
        formats = {im.format for im in self._region_images}
        if self.provides_mesh:
            formats = formats.union({"mesh"})
        if self.provides_volume:
            formats = formats.union({"volume"})
        return formats

    def get_filtered_map(self, regions: List[Region] = None) -> "Map":
        """
        Get the submap of this Map making up the regions provided.

        Regions that are parents of mapped chilren are provided, the children
        making up said regions (in this map) will be returned.
        """
        regionnames = [r.name for r in regions] if regions else None

        filtered_images = {
            regionname: attr_col
            for regionname, attr_col in self._index_mapping.items()
            if regionnames is None or regionname in regionnames
        }
        attributes = [
            Name(value=f"{regionnames} filtered from {self.name}"),
            _ID(value=None),
            self._get(SpeciesSpec),
        ]
        return replace(self, attributes=attributes, _index_mapping=filtered_images)

    def _find_images(self, regionname: str = None, frmt: str = None) -> List["Image"]:
        def filter_fn(im: "Image"):
            return im.of_format(frmt)

        if regionname is None:
            return list(filter(filter_fn, self._find(Image)))

        try:
            candidate = self.parcellation.get_region(regionname).name
            if candidate in self.regions:
                return list(
                    filter(filter_fn, self._index_mapping[candidate]._find(Image))
                )
        except AssertionError:
            pass

        # check if a parent region is requested
        candidates = {
            region.name
            for matched_region in self.parcellation.get_region(regionname)
            for region in matched_region.descendants
            if region.name in self.regions
        }
        if candidates:
            return [
                img
                for r in candidates
                for img in filter(
                    filter_fn,
                    self._index_mapping[r]._find(Image),
                )
            ]
        raise ValueError(
            f"Could not find any leaf or parent regions matching '{regionname}' in this map."
        )

    def fetch(
        self,
        region: Union[str, Region] = None,
        frmt: str = None,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        if isinstance(region, Region):
            regionspec = region.name
        else:
            regionspec = region
        if frmt is None:
            frmt = [
                f for f in IMAGE_FORMATS if f in self.image_formats - {"mesh", "volume"}
            ][0]
        else:
            assert (
                frmt in self.image_formats
            ), f"Requested format '{frmt}' is not available for this map: {self.image_formats=}."
        images = self._find_images(regionname=regionspec, frmt=frmt)

        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
        )

        if len(images) == 1:
            return images[0].fetch(**fetch_kwargs)

        if len(images) > 1:
            labels = [im.subimage_options["label"] for im in images]
            if set(labels) == {1}:
                labels = list(range(1, len(labels) + 1))
            return resample_to_template_and_merge(
                [img.fetch(**fetch_kwargs) for img in images], labels=labels
            )

        raise RuntimeError("No images found.")

    def get_colormap(self, frmt: str = None, regions: List[str] = None) -> List[str]:
        # TODO: profile and speed up
        from matplotlib.colors import ListedColormap

        def convert_hex_to_tuple(clr: str):
            return tuple(int(clr[p : p + 2], 16) for p in [1, 3, 5])

        if frmt is None:
            frmt = [f for f in IMAGE_FORMATS if f in self.image_formats][0]
        else:
            assert (
                frmt in self.image_formats
            ), f"Requested format '{frmt}' is not available for this map: {self.image_formats=}."

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
