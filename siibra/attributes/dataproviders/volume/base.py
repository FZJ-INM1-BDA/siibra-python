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

        # bbox = kwargs.pop("bbox", None)
        # resolution_mm = kwargs.pop("resolution_mm", None)
        # subpace = kwargs.pop("subpace", None)
        # labels = kwargs.pop("labels", None)
        # if self.format == "neuroglancer/precomputed":
        #     dataopsdict = neuroglancer.resolve_ops(
        #         bbox=bbox,
        #         resolution_mm=resolution_mm,
        #         subpace=subpace,
        #         labels=labels,
        #     )
        #     return super().get_data(**kwargs)

        # if subpace:
        #     self.transformation_ops.append(nifti.NiftiExtractSubspace(subpace))
        # if labels:
        #     self.transformation_ops.append(nifti.NiftiExtractLabels(labels))
        # # if resolution_mm:
        # #     self.transformation_ops.append(nifti.ResampleNifti(resolution_mm))
        # if bbox:
        #     self.transformation_ops.append(nifti.NiftiExtractVOI(bbox))
        # return super().get_data(**kwargs)
