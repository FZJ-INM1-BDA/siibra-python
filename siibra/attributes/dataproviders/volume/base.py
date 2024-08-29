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

from dataclasses import dataclass
from os import getenv
from typing import TYPE_CHECKING, TypedDict, Tuple, Dict

from ..base import DataProvider
from ....commons.iterable import assert_ooo

if TYPE_CHECKING:
    from ...locations import BoundingBox


SIIBRA_MAX_FETCH_SIZE_GIB = getenv("SIIBRA_MAX_FETCH_SIZE_GIB", 0.2)
IMAGE_FORMATS = ["nii", "neuroglancer/precomputed"]
MESH_FORMATS = ["gii-mesh", "gii-label", "freesurfer-annot", "neuroglancer/precompmesh"]
FORMAT_LOOKUP = {
    None: IMAGE_FORMATS + MESH_FORMATS,
    "mesh": MESH_FORMATS,
    "image": IMAGE_FORMATS,
}
READER_LOOKUP: Dict[str, Dict[str, str]] = {
    "nii": {"type": "read/nifti"},
}
# TODO: create the remaining readers and add to the dictionary


class Mapping(TypedDict):
    """
    Represents restrictions to apply to an image to get partial information,
    such as labelled mask, a specific slice etc.
    """

    label: int = None
    range: Tuple[float, float]
    subspace: Tuple[slice, ...]


class VolumeOpsKwargs(TypedDict):
    """
    Key word arguments used for fetching images and meshes across siibra.

    Note
    ----
    Not all parameters are avaialble for all formats and volumes.
    """

    bbox: "BoundingBox" = None
    resolution_mm: float = None
    max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB
    mapping: Dict[str, Mapping] = None


@dataclass
class VolumeProvider(DataProvider):
    schema: str = "siibra/attr/data/volume"
    space_id: str = None
    colormap: str = None  # TODO: remove from config and here

    def __post_init__(self):
        super().__post_init__()
        if self.format == "neuroglancer/precomputed":
            self.retrieval_ops.append(
                {"type": "read/neuroglancer_precomputed", "url": self.url}
            )
        elif self.format == "neuroglancer/precompmesh":
            self.retrieval_ops.append(
                {"type": "read/neuroglancer_precompmesh", "url": self.url}
            )
        elif self.format == "nii":
            self.transformation_ops.append({"type": "read/nifti"})
        elif self.format in {"gii-label", "gii-mesh"}:
            self.transformation_ops.append({"type": "read/gifti"})
        elif self.format in "freesurfer-annot":
            self.transformation_ops.append({"type": "read/freesurfer_annot"})

    @property
    def space(self):
        from ....factory import iter_preconfigured_ac
        from ....atlases import Space

        return assert_ooo(
            [
                space
                for space in iter_preconfigured_ac(Space)
                if space.ID == self.space_id
            ]
        )


def resolve_fetch_ops(vol_ops_kwargs: VolumeOpsKwargs):
    # if provider.format == "neuroglancer/precomputed":
    #     return [
    #         {
    #             "type": "read/neuroglancer_precomputed",
    #             "url": provider.url,
    #             "bbox": vol_ops_kwargs.get("bbox", None),
    #             "resolution_mm": vol_ops_kwargs.get("resolution_mm", None),
    #             "max_download_GB": vol_ops_kwargs.get(
    #                 "max_download_GB", SIIBRA_MAX_FETCH_SIZE_GIB
    #             ),
    #         }
    #     ]

    # if provider.format == "neuroglancer/precompmesh":
    #     return [{
    #         "type": "read/neuroglancer_precompmesh",
    #         "url": provider.url,
    #     }]

    ops = []
    bbox = vol_ops_kwargs.get("bbox")
    if bbox is not None:
        ops.append()

    resolution_mm = vol_ops_kwargs.get("resolution_mm")
    if resolution_mm is not None:
        # resolution_mm max_download_GB
        ops.append()

    color_channel = vol_ops_kwargs.get("color_channel")
    if color_channel is not None:
        ops.append()

    mapping = vol_ops_kwargs.get("mapping", None)
    if mapping is None:
        return ops

    subspace = mapping.get("subspace", None)
    if subspace is not None:
        ops.append({"tpye": "codec/vol/extractsubspace"})

    label = mapping.get("label", None)
    if label is not None:
        ops.append({"type": "codec/vol/extractlabels"})
    return ops
