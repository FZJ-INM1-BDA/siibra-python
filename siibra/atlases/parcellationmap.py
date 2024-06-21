from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, List, Dict, Set, Tuple

import numpy as np

from ..assignment import iter_attr_col
from ..concepts import AtlasElement
from ..retrieval_new.image_fetcher import FetchKwargs
from ..commons_new.iterable import assert_ooo
from ..commons_new.string import fuzzy_match
from ..commons_new.nifti_operations import resample_to_template_and_merge
from ..atlases import Parcellation, Space, Region
from ..dataitems import Image, IMAGE_FORMATS
from ..cache import fn_call_cache

from ..commons import SIIBRA_MAX_FETCH_SIZE_GIB, siibra_tqdm

if TYPE_CHECKING:
    from ..locations import BBox
    from ..concepts.attribute_collection import AttributeCollection
    from nibabel import Nifti1Image


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
                if parc.id == self.parcellation_id
            ]
        )

    @property
    def space(self) -> "Space":
        return assert_ooo([sp for sp in iter_attr_col(Space) if sp.id == self.space_id])

    @property
    def regions(self) -> Set[str]:
        return set(self._index_mapping.keys())

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

    def filter_regions(self, regions: List[Region] = None, format: str = None):
        regionnames = [r.name for r in regions] if regions else None
        filtered_images = {
            regionname: (
                attr_col.filter(
                    lambda attr: not isinstance(attr, Image) or attr.format == format
                )
                if format is not None
                else replace(attr_col)
            )
            for regionname, attr_col in self._index_mapping.items()
            if regionnames is None or regionname in regionnames
        }
        # TODO: Returning a new copy of this map with just indexmapping changed
        # is risky as attributes such as name and id will remain the same
        # because potentially we might use id and name for hashing or caching
        # purposes
        return replace(self, index_mapping=filtered_images)

    def _get_region(self, regionname: str):
        if regionname in self.regions:
            return regionname
        else:
            candidates = [r for r in self.regions if fuzzy_match(regionname, r)]
            if len(candidates) == 0:
                raise TypeError
            return assert_ooo(candidates)

    def _find_images(self, regionname: str = None, frmt: str = None) -> List["Image"]:
        def filter_fn(im: "Image"):
            return im.filter_format(frmt)

        if regionname is None:
            return list(filter(filter_fn, self._find(Image)))

        selected_region = self._get_region(regionname)

        return list(
            filter(
                filter_fn,
                self._index_mapping[selected_region]._find(Image),
            )
        )

    def fetch(
        self,
        regionname: str = None,
        frmt: str = None,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        if frmt is None:
            frmt = [
                f for f in IMAGE_FORMATS if f in self.image_formats - {"mesh", "volume"}
            ][0]
        else:
            assert (
                frmt in self.image_formats
            ), f"Requested format '{frmt}' is not available for this map: {self.image_formats=}."
        # TODO: reconsider `FetchKwargs`
        fetch_kwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
        )

        images = self._find_images(regionname=regionname, frmt=frmt)
        if len(images) == 1:
            return images[0].fetch(**fetch_kwargs)
        elif len(images) > 1:
            # TODO: must handle meshes
            # TODO: get header for affine and shape instead of the whole template
            template = self.space.get_template().fetch(**fetch_kwargs)
            # labels = [im.subimage_options["label"] for im in self._region_images]
            # if set(labels) == {1}:
            #     labels = list(range(1, len(labels) + 1))
            return resample_to_template_and_merge(
                [img.fetch(**fetch_kwargs) for img in images], template, labels=[]
            )
        else:
            raise RuntimeError("No images found.")

    def get_colormap(self, frmt: str = None, regions: List[str] = None) -> List[str]:
        from matplotlib.colors import ListedColormap
        import numpy as np

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


@dataclass
class SparseIndex:
    probs: List[float] = field(default_factory=list)
    bboxes: Dict = field(default_factory=dict)
    voxels: np.ndarray = field(default_factory=np.ndarray)
    affine: np.ndarray = field(default_factory=np.ndarray)
    shape: Tuple[int] = field(default_factory=tuple)

    def get_coords(self, regionname: str):
        # Nx3 array with x/y/z coordinates of the N nonzero values of the given mapindex
        coord_ids = [i for i, l in enumerate(self.probs) if regionname in l]
        x0, y0, z0 = self.bboxes[regionname]["minpoint"]
        x1, y1, z1 = self.bboxes[regionname]["maxpoint"]
        return (
            np.array(
                np.where(
                    np.isin(
                        self.voxels[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1],
                        coord_ids,
                    )
                )
            ).T
            + (x0, y0, z0)
        ).T

    def get_mapped_voxels(self, regionname: str):
        # returns the x, y, and z coordinates of nonzero voxels for the map
        # with the given index, together with their corresponding values v.
        x, y, z = [v.squeeze() for v in np.split(self.get_coords(regionname), 3)]
        v = [self.probs[i][regionname] for i in self.voxels[x, y, z]]
        return x, y, z, v

    def _exract_from_sparseindex(self, regionname: str):
        from nibabel import Nifti1Image

        x, y, z, v = self.get_mapped_voxels(regionname)
        result = np.zeros(self.shape, dtype=np.float32)
        result[x, y, z] = v
        return Nifti1Image(dataobj=result, affine=self.affine)


def add_img(spind: dict, nii: "Nifti1Image", regionname: str):
    imgdata = np.asanyarray(nii.dataobj)
    X, Y, Z = [v.astype("int32") for v in np.where(imgdata > 0)]
    for x, y, z, prob in zip(X, Y, Z, imgdata[X, Y, Z]):
        coord_id = spind["voxels"][x, y, z]
        if coord_id >= 0:
            # Coordinate already seen. Just add observed value.
            assert regionname not in spind["probs"][coord_id]
            assert len(spind["probs"]) > coord_id
            spind["probs"][coord_id][regionname] = prob
        else:
            # New coordinate. Append entry with observed value.
            coord_id = len(spind["probs"])
            spind["voxels"][x, y, z] = coord_id
            spind["probs"].append({regionname: prob})

    spind["bboxes"][regionname] = {
        "minpoint": (X.min(), Y.min(), Z.min()),
        "maxpoint": (X.max(), Y.max(), Z.max()),
    }
    return spind


@fn_call_cache
def build_sparse_index(parcmap: Map) -> SparseIndex:
    added_image_count = 0
    spind = {"voxels": {}, "probs": [], "bboxes": {}}
    mapaffine: np.ndarray = None
    mapshape: Tuple[int] = None
    for region, attrcol in siibra_tqdm(
        parcmap._index_mapping.items(),
        unit="map",
        desc=f"Building sparse index from {len(parcmap._index_mapping)} volumetric maps",
    ):
        image = attrcol._get(Image)
        nii = image.fetch()
        if added_image_count == 0:
            mapaffine = nii.affine
            mapshape = nii.shape
            spind["voxels"] = np.zeros(nii.shape, dtype=np.int32) - 1
        else:
            if (nii.shape != mapshape) or ((mapaffine - nii.affine).sum() != 0):
                raise RuntimeError(
                    "Building sparse maps from volumes with different voxel "
                    "spaces is not yet supported in siibra."
                )
        spind = add_img(spind, nii, region)
        added_image_count += 1
    return SparseIndex(
        probs=spind["probs"],
        bboxes=spind["bboxes"],
        voxels=spind["voxels"],
        affine=mapaffine,
        shape=mapshape,
    )


@dataclass
class SparseMap(Map):
    use_sparse_index: bool = False

    def __post_init__(self):
        super().__post_init__()

    @property
    def _sparse_index(self) -> SparseIndex:
        return build_sparse_index(self)

    def fetch(
        self,
        regionname: str = None,
        frmt: str = None,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ):
        selected_region = self._get_region(regionname)

        if self.use_sparse_index:
            nii = self._sparse_index._exract_from_sparseindex(selected_region)

        nii = super().fetch(regionname=selected_region, frmt=frmt)

        if bbox:
            from ..retrieval_new.image_fetcher.nifti import extract_voi

            nii = extract_voi(nii, bbox)

        if resolution_mm:
            from ..retrieval_new.image_fetcher.nifti import resample

            nii = resample(nii, resolution_mm)

        return nii
