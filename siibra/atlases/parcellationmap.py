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

from dataclasses import dataclass, replace, asdict, field
from typing import List, Set, Union, Dict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from pandas import DataFrame

from ..concepts import AtlasElement
from ..commons.iterable import assert_ooo
from ..commons.string import convert_hexcolor_to_rgbtuple
from ..commons.logger import logger, siibra_tqdm, QUIET
from ..atlases import ParcellationScheme, Space, Region
from ..attributes.dataproviders.volume import (
    VolumeProvider,
    ImageProvider,
    MeshProvider,
    VolumeOpsKwargs,
    Mapping,
    FORMAT_LOOKUP,
    IMAGE_FORMATS,
    MESH_FORMATS,
    SIIBRA_MAX_FETCH_SIZE_GIB,
)
from ..attributes.descriptions import Name, ID as _ID, SpeciesSpec
from ..attributes.locations import BoundingBox, Point, PointCloud
from ..operations.image_assignment import (
    ImageAssignment,
    ScoredImageAssignment,
    get_intersection_scores,
)
from ..operations.volume_fetcher.nifti import (
    MergeLabelledNiftis,
    NiftiMask,
    NiftiExtractLabels,
    NiftiExtractSubspace,
    NiftiExtractVOI,
)
from ..operations.base import Merge, DataOp

VALID_MAPTYPES = ("statistical", "labelled")


@dataclass(repr=False, eq=False)
class Map(AtlasElement):
    schema: str = "siibra/atlases/parcellationmap/v0.1"
    parcellation_id: str = None
    space_id: str = None
    maptype: Literal["labelled", "statistical"] = None
    region_mapping: Dict[str, List[Mapping]] = field(repr=False, default=None)

    def __post_init__(self):
        essential_specs = {"space_id", "maptype", "region_mapping"}
        assert all(
            self.__getattribute__(spec) is not None for spec in essential_specs
        ), f"Cannot create a parcellation `Map` without {essential_specs}"
        if len(self.regionnames) == 0:
            raise RuntimeError(f"Map does not contain any regions")
        super().__post_init__()

    @property
    def parcellation(self) -> "ParcellationScheme":
        from ..factory import iter_preconfigured_ac

        return assert_ooo(
            [
                parc
                for parc in iter_preconfigured_ac(ParcellationScheme)
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
    def regionnames(self) -> List[str]:
        return list(self.region_mapping.keys())

    @property
    def volume_providers(self) -> List[Union["ImageProvider", "MeshProvider"]]:
        return [
            attr
            for attr in self.attributes
            if isinstance(attr, (ImageProvider, MeshProvider))
        ]

    def decode_region(self, regionname: str):
        return self.parcellation.get_region(regionname)

    @property
    def formats(self) -> Set[str]:
        return {vol.format for vol in self.volume_providers}

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
        regionnames = [r.name if isinstance(r, Region) else r for r in regions]
        try:
            assert all(rn in self.regionnames for rn in regionnames)
        except AssertionError as e:
            raise ValueError(
                "Regions that are not mapped in this ParcellationMap are requested!"
            ) from e

        sub_mapping = {r: m for r, m in self.region_mapping.items() if r in regionnames}
        target_attrs = {mp["target"] for mps in sub_mapping.values() for mp in mps}
        vol_providers = [vp for vp in self.volume_providers if vp.name in target_attrs]
        attributes = [
            Name(value=f"{regionnames} filtered from {self.name}"),
            _ID(value=None),
            self._get(SpeciesSpec),
            *vol_providers,
        ]
        return replace(self, attributes=attributes, region_mapping=sub_mapping)

    def _select_format(self, frmt: Union[str, None] = None) -> str:
        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats}."
            )
        return frmt

    def _extract_regional_map_volume_provider(
        self,
        regionname: str,
        frmt: str = None,
    ):
        # N.B. region *must* be a leaf node (i.e. directly mapped, cannot be a parent)
        # TODO fix to allow parents?
        frmt = self._select_format(frmt)

        # TODO: filtering mappings should be smarter. ie what if there are two "volume/ref"?
        mappings = [
            m
            for m in self.region_mapping[regionname]
            if m.get("@type", "volume/ref") == "volume/ref"
        ]
        assert (
            len(mappings) == 1
        ), f"Expect one and only one mapping for {regionname}, but got {len(mappings)}"
        mapping = mappings[0]
        providers = [
            v
            for v in self.volume_providers
            if (
                v.format == frmt
                and (mapping.get("target") is None or v.name == mapping["target"])
            )
        ]
        if len(providers) == 0:
            raise RuntimeError(
                f"Expected at least one provider for {regionname}, but got 0."
            )

        if len(providers) == 1:
            provider = providers[0]
        else:
            logger.info(
                "Found several providers matching criteria. "
                + "\n".join([f" - {provider.format}" for provider in providers])
                + "Choosing the first."
            )
            provider = providers[0]

        transformpation_ops: List[DataOp] = []
        if mapping.get("label"):
            transformpation_ops.append(
                NiftiExtractLabels.generate_specs(labels=[mapping.get("label")])
            )
        if mapping.get("subspace"):
            transformpation_ops.append(
                NiftiExtractSubspace.generate_specs(subspace=mapping.get("subspace"))
            )

        return replace(
            provider, retrieval_ops=[], transformation_ops=transformpation_ops
        )

    def extract_regional_map(
        self,
        region: Union[str, Region, None] = None,
        frmt: str = None,
        bbox: "BoundingBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        regionname = None
        if isinstance(region, Region):
            regionname = region.name
        if isinstance(region, str):
            if region in self.regionnames:
                regionname = region
            else:
                regionname = self.parcellation.get_region(region).name
        if region is None:
            if len(self.regionnames) != 1:
                raise RuntimeError(
                    """Map contains multiple regions. Please provide one of the following as region:
                                   """
                    + "\n".join(self.regionnames)
                )
            regionname = self.regionnames[0]

        assert (
            regionname in self.regionnames
        ), f"{region} parsed to {regionname}, which was not found in regionnames"
        provider = self._extract_regional_map_volume_provider(
            regionname=regionname,
            frmt=frmt,
        )
        additional_transform_ops = []
        if bbox:
            additional_transform_ops.append(NiftiExtractVOI.generate_specs(voi=bbox))
        return replace(
            provider,
            transformation_ops=[
                *provider.transformation_ops,
                *additional_transform_ops,
            ],
        )

    def extract_mask(
        self,
        regions: List[Union[str, Region]],
        frmt: str = None,
        background_value: Union[int, float] = 0,
        lower_threshold: Union[int, float, None] = None,
    ):
        providers = [
            self._extract_regional_map_volume_provider(region, frmt=frmt)
            for region in regions
        ]
        provider_types = set(type(p) for p in providers)
        assert len(provider_types) == 1
        provider_type = next(iter(provider_types))
        mask_provider = provider_type(
            space_id=self.space_id,
            retrieval_ops=[
                Merge.generate_specs(
                    *[provider.retrieval_ops for provider in providers]
                )
            ],
            transformation_ops=[MergeLabelledNiftis.generate_specs()],
        )
        if isinstance(mask_provider, ImageProvider):
            if self.maptype == "statistical":
                mask_provider.transformation_ops.append(
                    NiftiMask.generate_specs(lower_threshold=lower_threshold)
                )
            else:
                mask_provider.transformation_ops.append(
                    NiftiMask.generate_specs(background_value=background_value)
                )
        else:
            # TODO: implement a gifti masker
            mask_provider.transformation_ops.append()

        return mask_provider.get_data()

    def extract_full_map(
        self,
        frmt: str = None,
        allow_relabeling: bool = False,
        as_binary_mask: bool = False
    ) -> VolumeProvider:
        """
        Extracts a single volume with all (sub)regions imcluded.
        """
        if as_binary_mask:
            return self.extract_mask(regions=self.regionnames)

        frmt = self._select_format(frmt)
        providers = [vp for vp in self.volume_providers if vp.format == frmt]
        assert len(set(type(p) for p in providers)) == 1
        fullmap_provider = providers[0].__class__(
            space_id=self.space_id,
            format=providers[0].format,
            retrieval_ops=[Merge.spec_from_dataproviders(providers)],
            transformation_ops=[MergeLabelledNiftis.generate_specs()],
        )

        if allow_relabeling:
            # TODO: create a relabeling dataop
            # fullmap_provider.append()
            # fullmap_provider.get_data()
            raise NotImplementedError("Re-labeling of maps is not yet implemented.")
        else:
            return fullmap_provider


    def get_colormap(self, regions: List[str] = None, frmt=None) -> List[str]:
        from matplotlib.colors import ListedColormap

        if frmt is None or frmt not in self.formats:
            frmt = [f for f in FORMAT_LOOKUP[frmt] if f in self.formats][0]
        else:
            assert frmt not in self.formats, RuntimeError(
                f"Requested format '{frmt}' is not available for this map: {self.formats}."
            )

        if regions is None:
            regions = self.regionnames

        assert all(
            r in self.regionnames for r in regions
        ), f"Please provide a subset of {self.regionnames}"

        label_color_table = {
            self.region_mapping[region]["label"]: convert_hexcolor_to_rgbtuple(
                self.region_mapping[region]["color"]
            )
            for region in regions
            for vol in self.volume_providers
            if vol.format == frmt and region in self.region_mapping
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

    def get_centroids(self, **volume_ops_kwargs: VolumeOpsKwargs) -> Dict[str, "Point"]:
        """
        Compute a dictionary of the centroids of all regions in this map.

        Returns
        -------
        Dict[str, Point]
            Region names as keys and computed centroids as items.
        """
        centroids = {}
        for regionname in siibra_tqdm(
            self.regionnames, unit="regions", desc="Computing centroids"
        ):
            img = self.extract_mask(
                region=regionname, **volume_ops_kwargs
            )  # returns a mask of the region
            centroid = compute_centroid(img)
            centroid.space_id = self.space_id
            centroids[regionname] = centroid
        return centroids

    # TODO: should be a dataop
    # def colorize(
    #     self, value_mapping: dict, **volume_ops_kwargs: VolumeOpsKwargs
    # ) -> "Nifti1Image":
    #     # TODO: rethink naming
    #     """
    #     Create

    #     Parameters
    #     ----------
    #     value_mapping : dict
    #         Dictionary mapping keys to values

    #     Return
    #     ------
    #     Nifti1Image
    #     """

    #     result = None
    #     nii = image.get_data(**volume_ops_kwargs)
    #     arr = np.asanyarray(nii.dataobj)
    #     resultarr = np.zeros_like(arr)
    #     result = nib.Nifti1Image(resultarr, nii.affine)
    #     for key, value in value_mapping.items():
    #         assert key in image.mapping, ValueError(
    #             f"key={key!r} is not in the mapping of the image."
    #         )
    #         resultarr[nii == image.mapping[key]["label"]] = value

    #     return result

    @dataclass
    class RegionAssignment(ImageAssignment):
        regionname: str

    @dataclass
    class ScoredRegionAssignment(ScoredImageAssignment):
        regionname: str

    def assign(
        self,
        queryitem: Union[Point, PointCloud, ImageProvider],
        split_components: bool = True,
        voxel_sigma_threshold: int = 3,
        iou_lower_threshold=0.0,
        statistical_map_lower_threshold: float = 0.0,
        **volume_ops_kwargs: VolumeOpsKwargs,
    ) -> DataFrame:
        assignments: List[Union[Map.RegionAssignment, Map.ScoredRegionAssignment]] = []
        for regionname in siibra_tqdm(self.regionnames, unit="region"):
            region_image = self._extract_regional_map_volume_provider(
                regionname=regionname, frmt="image", **volume_ops_kwargs
            )
            with QUIET:
                for assgnmt in get_intersection_scores(
                    queryitem=queryitem,
                    target_image=region_image,
                    split_components=split_components,
                    voxel_sigma_threshold=voxel_sigma_threshold,
                    iou_lower_threshold=iou_lower_threshold,
                    target_masking_lower_threshold=statistical_map_lower_threshold,
                    **volume_ops_kwargs,
                ):
                    if isinstance(assgnmt, ScoredImageAssignment):
                        assignments.append(
                            Map.ScoredRegionAssignment(
                                **asdict(assgnmt),
                                regionname=regionname,
                            )
                        )
                    else:
                        assignments.append(
                            Map.RegionAssignment(
                                **asdict(assgnmt),
                                regionname=regionname,
                            )
                        )

        return Map._convert_assignments_to_dataframe(assignments)

    @staticmethod
    def _convert_assignments_to_dataframe(assignments: List["Map.RegionAssignment"]):
        if any(isinstance(a, Map.ScoredRegionAssignment) for a in assignments):
            return DataFrame(
                assignments,
                columns=[
                    "intersection_over_union",
                    "intersection_over_first",
                    "intersection_over_second",
                    "correlation",
                    "weighted_mean_of_first",
                    "weighted_mean_of_second",
                    "map_value_mean",
                    "map_value_std",
                    "input_structure_index",
                    "centroid",
                    "map_value",
                    "regionname",
                ],
            ).dropna(axis="columns", how="all")
        else:
            return DataFrame(
                assignments,
                columns=[
                    "input_structure_index",
                    "centroid",
                    "map_value",
                    "regionname",
                ],
            )

    def lookup_points(
        self,
        points: Union[Point, PointCloud],
        **volume_ops_kwargs: VolumeOpsKwargs,
    ) -> DataFrame:
        points_ = (
            PointCloud.from_points([points]) if isinstance(points, Point) else points
        )
        if any(s not in {0.0} for s in points_.sigma):
            logger.warning(
                f"To get the full asignment score of uncertain points please use `{self.assign.__name__}`."
                "`lookup_points()` only considers the voxels the coordinates correspond to."
            )
            points_ = replace(points_, sigma=np.zeros(len(points_)).tolist())

        points_wrpd = points_.warp(self.space_id)

        assignments: List[Map.RegionAssignment] = []
        for region in siibra_tqdm(self.regionnames, unit="region"):
            region_image = self._extract_regional_map_volume_provider(
                regionname=region, frmt="image", **volume_ops_kwargs
            )
            for pointindex, map_value in zip(
                *region_image.lookup_points(points=points_wrpd, **volume_ops_kwargs)
            ):
                if map_value == 0:
                    continue
                assignments.append(
                    Map.RegionAssignment(
                        input_structure_index=pointindex,
                        centroid=points_[pointindex].coordinate,
                        map_value=map_value,
                        regionname=region,
                    )
                )
        return Map._convert_assignments_to_dataframe(assignments)
