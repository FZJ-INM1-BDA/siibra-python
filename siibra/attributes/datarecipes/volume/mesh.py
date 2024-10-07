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
from typing import Dict

from nibabel import GiftiImage

from .base import VolumeRecipe
from ...locations import BoundingBox
from ....operations.volume_fetcher.neuroglancer_precompmesh import ReadNgMeshes
from ....operations.volume_fetcher.gifti import MergeGifti, ReadGiftiFromBytesGii
from ....operations.volume_fetcher.freesurfer import ReadGiftiFromBytesFSAAnnot


def extract_label_mask(gii: GiftiImage, label: int):
    pass


@dataclass
class MeshRecipe(VolumeRecipe):
    schema: str = "siibra/attr/data/mesh/v0.1"

    @classmethod
    def _generate_ops(cls, conf: Dict):
        format = conf.get("format")
        if format == "neuroglancer/precompmesh":
            archive_options = conf.get("archive_options")
            label = archive_options.get("label")
            base_url = conf.get("url")
            assert label
            return [
                ReadNgMeshes.generate_specs(base_url=base_url, label=label),
                MergeGifti.generate_specs(),
            ]
        if format == "freesurfer-annot":
            return [
                *super()._generate_ops(conf),
                ReadGiftiFromBytesFSAAnnot.generate_specs(),
            ]
        if format == "gii-mesh" or format == "gii-label":
            return [*super()._generate_ops(), ReadGiftiFromBytesGii.generate_specs()]
        raise NotImplementedError(f"Does not know how to parse {format}")

    @property
    def boundingbox(self) -> "BoundingBox":
        raise NotImplementedError

    # def fetch(
    #     self,
    #     bbox: "BoundingBox" = None,
    #     resolution_mm: float = None,
    #     max_download_GB: float = SIIBRA_MAX_FETCH_SIZE_GIB,
    #     color_channel: int = None,
    # ) -> GiftiImage:
    #     fetchkwargs = VolumeOpsKwargs(
    #         bbox=bbox,
    #         resolution_mm=resolution_mm,
    #         color_channel=color_channel,
    #         max_download_GB=max_download_GB,
    #         mapping=self.mapping,
    #     )
    #     if color_channel is not None:
    #         assert self.format == "neuroglancer/precomputed"

    #     fetcher_fn = get_volume_fetcher(self.format)
    #     gii = fetcher_fn(self, fetchkwargs)

    #     mapping = fetchkwargs["mapping"]
    #     if mapping is not None and len(mapping) == 1:
    #         details = next(iter(mapping.values()))
    #         if "subspace" in details:
    #             s_ = tuple(
    #                 slice(None) if isinstance(s, str) else s
    #                 for s in details["subspace"]
    #             )
    #             gii = gii.slicer[s_]
    #         if "label" in details:
    #             gii = extract_label_mask(gii, details["label"])

    #     return gii

    def plot(self, *args, **kwargs):
        from nilearn import plotting

        return plotting.view_surf(
            self.get_data(),
            *args,
            **kwargs,
        )

    def get_data(self) -> GiftiImage:
        return super().get_data()
