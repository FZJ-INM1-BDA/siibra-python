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

from dataclasses import dataclass, replace, asdict
from typing import TYPE_CHECKING, List, Set, Union, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from ..concepts import AtlasElement
from ..retrieval.volume_fetcher import (
    FetchKwargs,
    IMAGE_FORMATS,
    MESH_FORMATS,
    SIIBRA_MAX_FETCH_SIZE_GIB,
)
from ..commons.iterable import assert_ooo
from ..commons.maps import merge_volumes, compute_centroid, create_mask
from ..commons.string import convert_hexcolor_to_rgbtuple
from ..commons.logger import logger, siibra_tqdm, QUIET
from ..atlases import Parcellation, Space, Region
from ..attributes.dataitems import Image, Mesh, FORMAT_LOOKUP
from ..attributes.descriptions import Name, ID as _ID, SpeciesSpec
from ..attributes.locations import BoundingBox, Point, PointCloud
from ..attributes.dataitems.volume.ops.intersection_score import (
    ImageAssignment,
    ScoredImageAssignment,
    get_intersection_scores,
)

if TYPE_CHECKING:
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
        from ..factory import iter_preconfigured_ac

        return assert_ooo(
            [
                parc
                for parc in iter_preconfigured_ac(Parcellation)
                if parc.ID == self.parcellation_id
            ]
        )

    @property
    def space(self) -> "Space":
        from ..factory import iter_preconfigured_ac

        return assert_ooo(
            [sp for sp in iter_preconfigured_ac(Space) if sp.ID == self.space_id]
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

    def get_label_mapping(
        self, regions: List[str] = None, frmt=None
    ) -> Dict[str, "Mapping"]:
        if regions is None:
            regions = self.regions
        else:
            assert all(
                r in self.regions for r in regions
            ), f"Please provide a subset of {self.regions}"

        if frmt in {None, "image", "mesh"}:
            frmt_ = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
            logger.debug(f"Selected format: {frmt!r}")
        elif frmt in self.formats:
            frmt_ = frmt
        else:
            raise RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats}."
            )

        return {
            key: val
            for vol in self.volumes
            for key, val in vol.mapping.items()
            if vol.format == frmt_ and vol.mapping is not None and key in regions
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
        regionnames = (
            [r.name for r in regions] if isinstance(regions[0], Region) else regions
        )
        try:
            assert all(rn in self.regions for rn in regionnames)
        except AssertionError:
            raise ValueError(
                "Regions that are not mapped in this ParcellationMap are requested!"
            )

        filtered_volumes = [
            replace(
                vol,
                mapping={r: vol.mapping[r] for r in regionnames if r in vol.mapping},
            )
            for vol in self.volumes
            if vol.mapping is not None and any(r in vol.mapping for r in regionnames)
        ]
        attributes = [
            Name(value=f"{regionnames} filtered from {self.name}"),
            _ID(value=None),
            self._get(SpeciesSpec),
            *filtered_volumes,
        ]
        return replace(self, attributes=attributes)

    def find_volumes(
        self, region: Union[str, Region] = None, frmt: str = None
    ) -> List[Union["Image", "Mesh"]]:
        if frmt in ["image", "mesh"]:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]

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
        mapped_descendants: Set[str] = {
            descendant.name
            for descendant in candidate.descendants
            if descendant.name in self.regions
        }
        if mapped_descendants:
            return [
                replace(
                    vol,
                    mapping={
                        desc: vol.mapping[desc]
                        for desc in mapped_descendants
                        if desc in vol.mapping
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
        bbox: "BoundingBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
        allow_relabeling: bool = False,
    ):
        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats}."
            )

        if isinstance(region, Region):
            regionspec = region.name
        else:
            regionspec = region

        volumes = self.find_volumes(region=regionspec, frmt=frmt)
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
                template_vol=self.space.fetch_template(**fetch_kwargs),
            )

    def fetch_mask(
        self,
        region: Union[str, Region] = None,
        background_value: Union[int, float] = 0,
        lower_threshold: float = None,
        frmt: str = None,
        bbox: "BoundingBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        volume = self.fetch(
            region=region,
            frmt=frmt,
            bbox=bbox,
            resolution_mm=resolution_mm,
            max_download_GB=max_download_GB,
            color_channel=color_channel,
        )
        return create_mask(
            volume, background_value=background_value, lower_threshold=lower_threshold
        )

    def get_colormap(self, regions: List[str] = None, frmt=None) -> List[str]:
        from matplotlib.colors import ListedColormap

        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats}."
            )

        if regions is None:
            regions = self.regions

        assert all(
            r in self.regions for r in regions
        ), f"Please provide a subset of {self.regions}"

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

    def get_centroids(self, **fetch_kwargs: FetchKwargs) -> Dict[str, "Point"]:
        """
        Compute a dictionary of the centroids of all regions in this map.

        Returns
        -------
        Dict[str, Point]
            Region names as keys and computed centroids as items.
        """
        centroids = {}
        for regionname in siibra_tqdm(
            self.regions, unit="regions", desc="Computing centroids"
        ):
            img = self.fetch(
                region=regionname, **fetch_kwargs
            )  # returns a mask of the region
            centroids[regionname] = compute_centroid(img, space_id=self.space)
        return centroids

    @dataclass
    class RegionAssignment(ImageAssignment):
        region: str

    @dataclass
    class ScoredRegionAssignment(ScoredImageAssignment):
        region: str

    def find_intersecting_regions(
        self,
        item: Union[Point, PointCloud, Image],
        split_components: bool = True,
        voxel_sigma_threshold: int = 3,
        iou_lower_threshold=0.0,
        statistical_map_lower_threshold: float = 0.0,
        **fetch_kwargs: FetchKwargs,
    ):
        from pandas import DataFrame

        assignments: List[Union[Map.RegionAssignment, Map.ScoredRegionAssignment]] = []
        for region in siibra_tqdm(self.regions, unit="region"):
            region_image = self.find_volumes(
                region=region, frmt="image", **fetch_kwargs
            )[0]
            with QUIET:
                for assgnmt in get_intersection_scores(
                    item=item,
                    target_image=region_image,
                    split_components=split_components,
                    voxel_sigma_threshold=voxel_sigma_threshold,
                    iou_lower_threshold=iou_lower_threshold,
                    statistical_map_lower_threshold=statistical_map_lower_threshold,
                    **fetch_kwargs,
                ):
                    if isinstance(assgnmt, ScoredImageAssignment):
                        assgnmt_type = Map.ScoredRegionAssignment
                    else:
                        assgnmt_type = Map.RegionAssignment
                    assignments.append(assgnmt_type(**asdict(assgnmt), region=region))

        assignments_unpacked = [asdict(a) for a in assignments]

        return (
            DataFrame(assignments_unpacked)
            .convert_dtypes()  # convert will guess numeric column types
            .dropna(axis="columns", how="all")
        )
