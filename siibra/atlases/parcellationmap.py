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
from collections import defaultdict

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
from pandas import DataFrame

from ..exceptions import SiibraException
from ..concepts import atlas_elements, query_parameter
from ..commons.iterable import assert_ooo
from ..commons.string import convert_hexcolor_to_rgbtuple
from ..commons.logger import logger, siibra_tqdm, QUIET
from ..atlases import ParcellationScheme, Space, Region
from ..attributes.datarecipes.volume import (
    VolumeRecipe,
    ImageRecipe,
    VolumeOpsKwargs,
    Mapping,
)
from ..attributes.descriptions import Name, ID as _ID, SpeciesSpec
from ..attributes.locations import BoundingBox, Point, PointCloud
from ..operations.base import Merge
from ..operations.image_assignment import (
    ImageAssignment,
    ScoredImageAssignment,
    get_intersection_scores,
)
from ..operations.volume_fetcher import VolumeFormats
from ..operations.volume_fetcher.nifti import (
    MergeLabelledNiftis,
    NiftiMask,
    NiftiExtractLabels,
    NiftiExtractSubspace,
    NiftiExtractVOI,
)
from ..operations.base import Merge

VALID_MAPTYPES = ("statistical", "labelled")


@dataclass(repr=False, eq=False)
class Map(atlas_elements.AtlasElement):
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
            raise RuntimeError("Map does not contain any regions")
        super().__post_init__()

    @property
    def parcellation(self) -> "ParcellationScheme":
        from .. import find

        query_param = query_parameter.QueryParam(
            attributes=[_ID(value=self.parcellation_id)]
        )
        return assert_ooo(find([query_param], ParcellationScheme))

    @property
    def space(self) -> "Space":
        from .. import find

        query_param = query_parameter.QueryParam(attributes=[_ID(value=self.space_id)])
        return assert_ooo(find([query_param], Space))

    @property
    def regionnames(self) -> List[str]:
        return list(self.region_mapping.keys())

    def get_region(self, regionname: str) -> Region:
        return self.parcellation.get_region(regionname)

    @property
    def formats(self) -> Set[str]:
        return {vol.format for vol in self.volume_recipes}

    @property
    def provides_mesh(self):
        # TODO (2.1) use VolumeFormats.Category
        return any(f in self.formats for f in VolumeFormats.MESH_FORMATS)

    @property
    def provides_image(self):
        # TODO (2.1) use VolumeFormats.Category
        return any(f in self.formats for f in VolumeFormats.IMAGE_FORMATS)

    def _select_format(self, frmt: Union[str, None] = None) -> str:

        if frmt in self.formats:
            return frmt

        if frmt in VolumeFormats.Category:
            for category_format in VolumeFormats.FORMAT_LOOKUP[frmt]:
                if category_format in self.formats:
                    return category_format
            raise SiibraException(
                f"Unable to find a suitable format from category {frmt}"
            )

        if frmt is None:

            for lookup_format in VolumeFormats.FORMAT_LOOKUP[None]:
                if lookup_format in self.formats:
                    return lookup_format
            raise SiibraException(
                f"Unable to find a supported format from {self.formats}. Most likely, "
                "this is due to readers not registered to read the formats."
            )

        raise SiibraException(
            """
            Unable to find a suitable format. Please either specify the category: 
            """
            + "\n".join(f" - {category.value}" for category in VolumeFormats.Category)
            + """
            Or one of the format directly:
            """
            + "\n".join(f" - {f}" for f in self.formats)
        )

    def _extract_regional_map_volume_recipe(
        self,
        regionname: str,
        frmt: str = None,
    ):
        # N.B. region *must* be a leaf node (i.e. directly mapped, cannot be a parent)
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
            for v in self.volume_recipes
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

        extraction_ops: List[Dict] = []
        if mapping.get("label"):
            extraction_ops.append(
                NiftiExtractLabels.generate_specs(labels=[mapping.get("label")])
            )
        if mapping.get("subspace"):
            extraction_ops.append(
                NiftiExtractSubspace.generate_specs(subspace=mapping.get("subspace"))
            )

        return replace(provider, _ops=[*provider._ops, *extraction_ops])

    def _find_region_mappings(self, region: Region) -> List[Mapping]:
        """
        Returns List of Mapping from a region object. If the region is not directly mapped, will iterate over its children, and flatten the returned List.

        Caller should do any sanitation/check.
        """

        # if region is already in mappings, return the mappings
        if region.name in self.region_mapping:
            return self.region_mapping.get(region.name) or []

        # if not, iterate over all children, flatten the mappings into a single list, return
        return [
            cmapping
            for c in region
            if c is not region
            for cmapping in self._find_region_mappings(c)
        ]

    def extract_regional_map(
        self, region: Union[str, Region, None] = None, format: str = None
    ):
        """
        Build a image recipe based on region/format provided. Providing a region instance rather than region name is more performant, and preferred.
        """

        selected_format = self._select_format(format)

        if region is None:
            region = self.parcellation
        if isinstance(region, str):
            region = self.parcellation.get_region(region)

        all_mappings = self._find_region_mappings(region)
        volume_mapping = [
            m for m in all_mappings if m.get("@type", "volume/ref") == "volume/ref"
        ]

        target_volumemapping: Dict[str, List[Mapping]] = defaultdict(list)
        for mapping in volume_mapping:
            target_volumemapping[mapping.get("target")].append(mapping)

        recipes: List[ImageRecipe] = []
        for volrecipe in self.volume_recipes:
            if volrecipe.format != selected_format:
                continue
            if volrecipe.name not in target_volumemapping:
                continue
            mappings = target_volumemapping[volrecipe.name]
            labels = [m["label"] for m in mappings]
            recipes.append(volrecipe.reconfigure(labels=labels))

        if len(recipes) == 0:
            raise SiibraException(f"No Image recipe found")
        if len(recipes) == 1:
            return recipes[0]

        # TODO (2.0) what happens if Map is provides for Mesh instead of Image?
        return ImageRecipe(
            _ops=[
                Merge.spec_from_datarecipes(recipes),
                MergeLabelledNiftis.generate_specs(),
            ],
            format="nii",
            space_id=self.space_id,
        )

    def extract_mask(
        self,
        regions: List[Union[str, Region]],
        frmt: str = None,
        background_value: Union[int, float] = 0,
        lower_threshold: Union[int, float, None] = None,
    ):
        frmt_ = self._select_format(frmt)
        recipes = [
            self._extract_regional_map_volume_recipe(region, frmt=frmt_)
            for region in regions
        ]
        assert (
            len(recipes) > 0
        ), f"Did not find any volume recipe with the given parameters: {regions}, {frmt}"

        mask_recipe = None
        if len(recipes) == 1:
            mask_recipe = recipes[0]
        else:
            provider_types = set(type(p) for p in recipes)

            assert len(provider_types) == 1
            provider_type = next(iter(provider_types))

            mask_recipe = provider_type(
                space_id=self.space_id,
                format=frmt_,
                _ops=[
                    Merge.spec_from_datarecipes(recipes),
                    MergeLabelledNiftis.generate_specs(),
                ],
            )

        if isinstance(mask_recipe, ImageRecipe):
            if self.maptype == "statistical":
                mask_recipe._ops.append(
                    NiftiMask.generate_specs(lower_threshold=lower_threshold)
                )
            else:
                mask_recipe._ops.append(
                    NiftiMask.generate_specs(background_value=background_value)
                )
        else:
            # TODO: implement a gifti masker
            raise NotImplementedError
            # mask_recipe.override_ops.append()

        return mask_recipe

    def extract_full_map(
        self,
        frmt: str = None,
        allow_relabeling: bool = False,
        as_binary_mask: bool = False,
    ) -> VolumeRecipe:
        """
        Extracts a single volume with all (sub)regions included.
        """

        if as_binary_mask:
            return self.extract_mask(regions=self.regionnames)

        frmt = self._select_format(frmt)
        providers = [vp for vp in self.volume_recipes if vp.format == frmt]
        assert len(set(type(p) for p in providers)) == 1

        labels = list(range(len(self.regionnames))) if allow_relabeling else []

        fullmap_recipe = providers[0].__class__(
            space_id=self.space_id,
            format=providers[0].format,
            _ops=[
                Merge.spec_from_datarecipes(providers),
                MergeLabelledNiftis.generate_specs(labels=labels),
            ],
        )

        return fullmap_recipe

    def get_colormap(self, regions: List[str] = None, frmt=None) -> List[str]:
        from matplotlib.colors import ListedColormap

        if frmt is None or frmt not in self.formats:
            frmt = [f for f in VolumeFormats.FORMAT_LOOKUP[frmt] if f in self.formats][
                0
            ]
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
            mapping["label"]: convert_hexcolor_to_rgbtuple(mapping["color"])
            for region in regions
            for vol in self.volume_recipes
            for mapping in self.region_mapping[region]
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

    # def get_centroids(self, **volume_ops_kwargs: VolumeOpsKwargs) -> Dict[str, "Point"]:
    #     """
    #     Compute a dictionary of the centroids of all regions in this map.

    #     Returns
    #     -------
    #     Dict[str, Point]
    #         Region names as keys and computed centroids as items.
    #     """
    #     centroids = {}
    #     for regionname in siibra_tqdm(
    #         self.regionnames, unit="regions", desc="Computing centroids"
    #     ):
    #         img = self.extract_mask(
    #             region=regionname, **volume_ops_kwargs
    #         )  # returns a mask of the region
    #         centroid = compute_centroid(img)
    #         centroid.space_id = self.space_id
    #         centroids[regionname] = centroid
    #     return centroids

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
        queryitem: Union[Point, PointCloud, ImageRecipe],
        split_components: bool = True,
        voxel_sigma_threshold: int = 3,
        iou_lower_threshold=0.0,
        statistical_map_lower_threshold: float = 0.0,
        **volume_ops_kwargs: VolumeOpsKwargs,
    ) -> DataFrame:
        if isinstance(queryitem, Point) and queryitem.sigma == 0:
            return self.lookup_points(queryitem, **volume_ops_kwargs)
        if isinstance(queryitem, PointCloud):
            sigmas = set(queryitem.sigma)
            if len(sigmas) == 1 and 0 in sigmas:
                return self.lookup_points(queryitem, **volume_ops_kwargs)

        assignments: List[Union[Map.RegionAssignment, Map.ScoredRegionAssignment]] = []
        for regionname in siibra_tqdm(self.regionnames, unit="region"):
            region_image = self._extract_regional_map_volume_recipe(
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

    # TODO (ASAP) broken if parcellation map is neuroglancer precomputed.
    # *should* be fixed by the new DataRecipe paradigm
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
            region_image = self._extract_regional_map_volume_recipe(
                regionname=region, frmt="image", **volume_ops_kwargs
            )
            assert isinstance(region_image, ImageRecipe)
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
