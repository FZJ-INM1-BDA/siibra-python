# Copyright 2018-2025
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
"""Multimodal data features in 2D section."""

from . import image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...locations import AxisAlignedPatch
    from ...features.anchor import AnatomicalAnchor


class CellbodyStainedSection(
    image.Image,
    configuration_folder="features/images/sections/cellbody",
    category="cellular",
):
    def __init__(self, **kwargs):
        image.Image.__init__(self, **kwargs, modality="cell body staining")


class BigBrain1MicronPatch(image.Image, category="cellular"):
    def __init__(
        self,
        patch: "AxisAlignedPatch",
        section: CellbodyStainedSection,
        relevance: float,
        anchor: "AnatomicalAnchor",
        **kwargs
    ):
        self._patch = patch
        self._section = section
        self.relevance = relevance
        image.Image.__init__(
            self,
            name=f"Cortical patch in {section.name}",
            modality=section.modality,
            space_spec=section._space_spec,
            providers=list(section._providers.values()),
            region=None,
            datasets=section.datasets,
            bbox=patch.boundingbox,
            id=None
        )
        self._anchor_cached = anchor

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(space_spec={self._space_spec}, "
            f"name='{self.name}', patch='{self._patch}', providers={self._providers})>"
        )

    def fetch(self, flip=False, **kwargs):
        assert "voi" not in kwargs
        res = kwargs.get("resolution_mm", -1)
        if flip:
            return self._patch.flip().extract_volume(
                self._section, resolution_mm=res
            )
        else:
            return self._patch.extract_volume(
                self._section, resolution_mm=res
            )
