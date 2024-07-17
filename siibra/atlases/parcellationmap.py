# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List, Set, Union, Literal, Dict

import numpy as np

from ..concepts import AtlasElement
from ..retrieval.volume_fetcher import FetchKwargs, IMAGE_FORMATS, MESH_FORMATS
from ..commons_new.iterable import assert_ooo
from ..commons_new.maps import merge_volumes
from ..commons_new.string import convert_hexcolor_to_rgbtuple
from ..commons_new.logger import logger
from ..atlases import Parcellation, Space, Region
from ..attributes.dataitems import Image, Mesh, FORMAT_LOOKUP
from ..attributes.descriptions import Name, ID as _ID, SpeciesSpec

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB

if TYPE_CHECKING:
    from ..attributes.locations import BBox
    from ..retrieval.volume_fetcher import Mapping


VALID_MAPTYPES = ("statistical", "labelled")


@dataclass(repr=False, eq=False)
class Map(AtlasElement):
    schema: str = "siibra/atlases/parcellationmap/v0.1"
    parcellation_id: str = None
    space_id: str = None
    maptype: Literal["labelled", "statistical"] = None

    def __post_init__(self):
        essential_specs = {
            "space_id",
            "maptype",
        }
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

        return assert_ooo(
            [sp for sp in iter_collection(Space) if sp.ID == self.space_id]
        )

    @property
    def regions(self) -> List[str]:
        return list(
            dict.fromkeys(
                key
                for vol in self.volumes
                for key in vol.mapping.keys()
                if vol.mapping is not None
            )
        )

    def get_mapping(self, frmt=None, regions: List[str] = None) -> Dict[str, "Mapping"]:
        if regions is None:
            regions = self.regions

        assert all(
            r in self.regions for r in regions
        ), f"Please provide a subset of {self.regions=}"

        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
            logger.info(f"Selected format: {frmt=!r}")
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats=}."
            )

        return {
            key: val
            for vol in self.volumes for key, val in vol.mapping.items()
            if vol.mapping is not None and key in regions
        }

    @property
    def volumes(self) -> List[Union["Image", "Mesh"]]:
        return [attr for attr in self.attributes if isinstance(attr, (Image, Mesh))]

    @property
    def formats(self) -> Set[str]:
        return {vol.format for vol in self.volumes}

    @property
    def provides_mesh(self):
        return any(f in self.formats for f in MESH_FORMATS)

    @property
    def provides_image(self):
        return any(f in self.formats for f in IMAGE_FORMATS)

    def get_filtered_map(self, regions: Union[List[Region], List[str]]) -> "Map":
        """
        Get the submap of this Map making up the regions provided.

        Raises
        ----
        ValueError
            If any region not mapped in this map is requested.
        """
        regionnames = [r.name for r in regions] if isinstance(regions[0], Region) else regions
        try:
            assert all(rn in self.regions for rn in regionnames)
        except AssertionError:
            raise ValueError("Regions that are not mapped in this ParcellationMap are requested!")

        filtered_volumes = [
            replace(vol, mapping={
                r: vol.mapping[r]
                for r in regionnames
                if r in vol.mapping
            })
            for vol in self.volumes if vol.mapping is not None and any(r in vol.mapping for r in regionnames)
        ]
        attributes = [
            Name(value=f"{regionnames} filtered from {self.name}"),
            _ID(value=None),
            self._get(SpeciesSpec),
            *filtered_volumes
        ]
        return replace(self, attributes=attributes)

    def find_volumes(
        self, region: Union[str, Region] = None, frmt: str = None
    ) -> List[Union["Image", "Mesh"]]:
        def filter_format(vol: Union["Image", "Mesh"]):
            return True if frmt is None else vol.format == frmt

        if region is None:
            return list(filter(filter_format, self.volumes))

        _regionname = region.name if isinstance(region, Region) else region

        if _regionname in self.regions:
            return [
                replace(vol, mapping={_regionname: vol.mapping[_regionname]})
                for vol in self.volumes
                if filter_format(vol) and _regionname in vol.mapping
            ]

        candidate = self.parcellation.get_region(_regionname)
        if candidate.name in self.regions:
            return [
                replace(vol, mapping={candidate.name: vol.mapping[candidate.name]})
                for vol in self.volumes
                if filter_format(vol) and candidate.name in vol.mapping
            ]

        # check if a parent region is requested
        mapped_descendants = {
            descendant.name
            for descendant in candidate.descendants
            if descendant.name in self.regions
        }
        if mapped_descendants:
            return [
                replace(
                    vol,
                    mapping={
                        desc.name: vol.mapping[desc.name]
                        for desc in mapped_descendants
                        if desc.name in vol.mapping
                    },
                )
                for vol in filter(
                    filter_format,
                    self.volumes,
                )
            ]

        raise ValueError(
            f"Could not find any leaf or parent regions matching '{_regionname}' in this map."
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
    ):
        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats=}."
            )

        if isinstance(region, Region):
            regionspec = region.name
        else:
            regionspec = region

        volumes = self.find_volumes(_regionname=regionspec, frmt=frmt)
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
            labels = [mp["label"] for vol in volumes for mp in vol.mapping.values()]
            if set(labels) == {1}:
                labels = list(range(1, len(labels) + 1))

        if frmt in MESH_FORMATS:
            logger.debug("Merging mesh labels.")
            return merge_volumes(
                [vol.fetch(**fetch_kwargs) for vol in volumes],
                labels=labels,
            )

        niftis = [vol.fetch(**fetch_kwargs) for vol in volumes]
        shapes = set(v.shape for v in niftis)
        try:
            assert len(shapes) == 1
            return merge_volumes(
                niftis,
                labels=labels,
            )
        except AssertionError:
            return merge_volumes(
                niftis,
                labels=labels,
                template_vol=self.space.fetch_template(**fetch_kwargs)
            )

    def get_colormap(self, regions: List[str] = None, frmt=None) -> List[str]:
        from matplotlib.colors import ListedColormap

        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats=}."
            )

        if regions is None:
            regions = self.regions

        assert all(
            r in self.regions for r in regions
        ), f"Please provide a subset of {self.regions=}"

        label_color_table = {
            vol.mapping[region]["label"]: convert_hexcolor_to_rgbtuple(
                vol.mapping[region]["color"]
            )
            for region in regions
            for vol in self.volumes
            if vol.format == frmt and region in vol.mapping
        }
        pallette = np.array(
            [
                (
                    list(label_color_table[i]) + [1]
                    if i in label_color_table
                    else [0, 0, 0, 0]
                )
                for i in range(max(label_color_table.keys()) + 1)
            ]
        ) / [255, 255, 255, 1]
        return ListedColormap(pallette)
