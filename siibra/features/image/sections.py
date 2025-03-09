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

from typing import TYPE_CHECKING

from . import image

if TYPE_CHECKING:
    from ...locations import AxisAlignedPatch, Contour
    from ...features.anchor import AnatomicalAnchor


class CellbodyStainedSection(
    image.Image,
    configuration_folder="features/images/sections/cellbody",
    category="cellular",
):
    def __init__(self, **kwargs):
        image.Image.__init__(self, **kwargs, modality="cell body staining")


class BigBrain1MicronPatch(image.Image, category="cellular"):

    _DESCRIPTION = """Sample approximately orthogonal cortical image patches
    from BigBrain 1 micron sections, guided by an image volume
    in a supported reference space providing. The image
    volume is used as a weighted mask to extract patches
    along the cortical midsurface with nonzero weights in the
    input image.
    An optional lower_threshold can be used to narrow down
    the search. The weight is stored with the resulting features."""

    def __init__(
        self,
        patch: "AxisAlignedPatch",
        profile: "Contour",
        section: CellbodyStainedSection,
        vertex: int,
        relevance: float,
        anchor: "AnatomicalAnchor",
    ):
        self._patch = patch
        self._profile = profile
        self._section = section
        self.vertex = vertex
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
            id=None,
        )
        self._anchor_cached = anchor
        self._description_cached = self._DESCRIPTION

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}(space_spec={self._space_spec}, "
            f"name='{self.name}', "
            f"section='{self._section.get_boundingbox().minpoint.bigbrain_section()}', "
            f"vertex='{self.vertex}', providers={self._providers})>"
        )

    @property
    def section(self) -> CellbodyStainedSection:
        return self._section

    def get_boundingbox(self):
        """ Enforce that the bounding box spans the full section thickness."""
        bbox_section = self._section.get_boundingbox()
        bbox = self._patch.boundingbox
        bbox.minpoint[1] = bbox_section.minpoint[1]
        bbox.maxpoint[1] = bbox_section.maxpoint[1]
        return bbox

    @property
    def profile(self) -> "Contour":
        return self._profile

    @property
    def bigbrain_section(self):
        return self.get_boundingbox().minpoint.bigbrain_section()

    def fetch(self, flip: bool = False, resolution_mm: float = -1, **kwargs):
        assert len(kwargs) == 0
        p = self._patch.flip() if flip else self._patch
        return p.extract_volume(self._section, resolution_mm=resolution_mm).fetch()
