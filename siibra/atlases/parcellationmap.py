from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Dict, Set, Union, Literal

import numpy as np

from ..concepts import AtlasElement
from ..retrieval_new.volume_fetcher import FetchKwargs, IMAGE_FORMATS, MESH_FORMATS
from ..commons_new.iterable import assert_ooo
from ..commons_new.maps import merge_volumes
from ..atlases import Parcellation, Space, Region
from ..dataitems import Image, Mesh
from ..descriptions import Name, ID as _ID, SpeciesSpec

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ..locations import BBox
    from ..concepts.attribute_collection import AttributeCollection


VALID_MAPTYPES = ("statistical", "labelled")


@dataclass(repr=False, eq=False)
class Map(AtlasElement):
    schema: str = "siibra/atlases/parcellation_map/v0.1"
    parcellation_id: str = None
    space_id: str = None
    maptype: Literal["labelled", "statistical"] = None
    _region_attributes: Dict[str, "AttributeCollection"] = field(default_factory=dict)

    def __post_init__(self):
        essential_specs = {"parcellation_id", "space_id", "maptype", "_region_attributes"}
        assert all(
            self.__getattribute__(spec) is not None for spec in essential_specs
        ), f"Cannot create a parcellation `Map` without {essential_specs}"
        super().__post_init__()

    @property
    def parcellation(self) -> "Parcellation":
        from ..factory import iter_collection
        return assert_ooo(
            [
                parc
                for parc in iter_collection(Parcellation)
                if parc.ID == self.parcellation_id
            ]
        )

    @property
    def space(self) -> "Space":
        from ..factory import iter_collection
        return assert_ooo([sp for sp in iter_collection(Space) if sp.ID == self.space_id])

    @property
    def regions(self) -> List[str]:
        return list(self._region_attributes.keys())

    @property
    def _region_volumes(self) -> List[Union["Image", "Mesh"]]:
        return [
            attr
            for attrcols in self._region_attributes.values()
            for attr in attrcols.attributes
            if isinstance(attr, (Image, Mesh))
        ]

    @property
    def volumes(self):
        return [
            attr for attr in self.attributes if isinstance(attr, (Mesh, Image))
        ] + self._region_volumes

    @property
    def formats(self) -> Set[str]:
        return {vol.format for vol in self._region_volumes}

    @property
    def provides_mesh(self):
        return any(f in self.formats for f in MESH_FORMATS)

    @property
    def provides_image(self):
        return any(f in self.formats for f in IMAGE_FORMATS)

    def get_filtered_map(self, regions: List[Region] = None) -> "Map":
        """
        Get the submap of this Map making up the regions provided.

        Regions that are parents of mapped chilren are provided, the children
        making up said regions (in this map) will be returned.
        """
        regionnames = [r.name for r in regions] if regions else None

        filtered_images = {
            regionname: attr_col
            for regionname, attr_col in self._region_attributes.items()
            if regionnames is None or regionname in regionnames
        }
        attributes = [
            Name(value=f"{regionnames} filtered from {self.name}"),
            _ID(value=None),
            self._get(SpeciesSpec),
        ]
        return replace(self, attributes=attributes, _index_mapping=filtered_images)

    def _find_volumes(
        self, regionname: str = None, frmt: str = None
    ) -> List[Union["Image", "Mesh"]]:
        def filter_fn(vol: Union["Image", "Mesh"]):
            return True if frmt is None else vol.format == frmt

        if regionname is None:
            return [
                attr
                for attr in self.attributes
                if isinstance(attr, (Mesh, Image)) and filter_fn(attr)
            ]

        try:
            candidate = self.parcellation.get_region(regionname).name
            if candidate in self.regions:
                return [
                    attr
                    for attr in self._region_attributes[candidate].attributes
                    if isinstance(attr, (Mesh, Image)) and filter_fn(attr)
                ]

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
                vol
                for c in candidates
                for vol in filter(
                    filter_fn,
                    self._region_attributes[c]._find(Image),
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
        allow_relabeling: bool = False,
        variant: str = None,
    ):
        if isinstance(region, Region):
            regionspec = region.name
        else:
            regionspec = region

        if frmt not in self.formats:
            frmt_lookup = {
                None: IMAGE_FORMATS + MESH_FORMATS,
                "mesh": MESH_FORMATS,
                "image": IMAGE_FORMATS,
            }
            try:
                frmt = [f for f in frmt_lookup[frmt] if f in self.formats][0]
            except KeyError:
                raise RuntimeError(
                    f"Requested format '{frmt}' is not available for this map: {self.formats=}."
                )

        volumes = self._find_volumes(regionname=regionspec, frmt=frmt)
        if len(volumes) == 0:
            raise RuntimeError("No images or meshes found matching parameters.")

        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
        )

        if len(volumes) == 1:
            return volumes[0].fetch(**fetch_kwargs)

        labels = []
        if allow_relabeling:
            labels = [vol.volume_selection_options["label"] for vol in volumes]
            if set(labels) == {1}:
                labels = list(range(1, len(labels) + 1))
        if variant:
            tmp = self.space.fetch_template(frmt="mesh", variant=variant)
            return merge_volumes(
                [vol.fetch(**fetch_kwargs) for vol in volumes],
                labels=labels,
                template_vol=tmp,
            )
        return merge_volumes(
            [vol.fetch(**fetch_kwargs) for vol in volumes],
            labels=labels,
        )

    def get_colormap(self, frmt: str = None, regions: List[str] = None) -> List[str]:
        # TODO: profile and speed up
        from matplotlib.colors import ListedColormap

        def convert_hex_to_tuple(clr: str):
            return tuple(int(clr[p : p + 2], 16) for p in [1, 3, 5])

        if frmt not in self.formats:
            frmt_lookup = {
                None: IMAGE_FORMATS + MESH_FORMATS,
                "mesh": MESH_FORMATS,
                "image": IMAGE_FORMATS,
            }
            try:
                frmt = [f for f in frmt_lookup[frmt] if f in self.formats][0]
            except KeyError:
                raise RuntimeError(
                    f"Requested format '{frmt}' is not available for this map: {self.formats=}."
                )

        if regions is None:
            regions = self.regions

        colors = {
            im.volume_selection_options["label"]: convert_hex_to_tuple(im.color)
            for region in regions
            for im in self._find_volumes(region, frmt=frmt)
        }
        pallette = np.array(
            [
                list(colors[i]) + [1] if i in colors else [0, 0, 0, 0]
                for i in range(max(colors.keys()) + 1)
            ]
        ) / [255, 255, 255, 1]
        return ListedColormap(pallette)
