from dataclasses import dataclass

from nibabel import GiftiImage

from ....commons import SIIBRA_MAX_FETCH_SIZE_GIB

from .base import Volume
from ...locations import BBox
from ....retrieval_new.volume_fetcher.volume_fetcher import (
    get_volume_fetcher,
    get_bbox_getter,
    FetchKwargs,
    MESH_FORMATS,
)


def extract_label_mask(gii: GiftiImage, label: int):
    pass


@dataclass
class Mesh(Volume):
    schema: str = "siibra/attr/data/mesh/v0.1"

    def __post_init__(self):
        assert self.format in MESH_FORMATS

    @property
    def boundingbox(self) -> "BBox":
        bbox_getter = get_bbox_getter(self.format)
        return bbox_getter(self)

    def fetch(
        self,
        bbox: "BBox" = None,
        resolution_mm: float = None,
        max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
        color_channel: int = None,
    ) -> GiftiImage:
        fetchkwargs = FetchKwargs(
            bbox=bbox,
            resolution_mm=resolution_mm,
            color_channel=color_channel,
            max_download_GB=max_download_GB,
            mapping=self.mapping
        )
        if color_channel is not None:
            assert self.format == "neuroglancer/precomputed"

        fetcher_fn = get_volume_fetcher(self.format)
        gii = fetcher_fn(self, fetchkwargs)

        mapping = fetchkwargs["mapping"]
        if mapping is not None and len(mapping) == 1:
            details = next(iter(mapping.values()))
            if "subspace" in details:
                s_ = tuple(
                    slice(None) if isinstance(s, str) else s for s in details["subspace"]
                )
                gii = gii.slicer[s_]
            if "label" in details:
                gii = extract_label_mask(gii, details["label"])

        return gii

    def plot(self, *args, **kwargs):
        raise NotImplementedError
