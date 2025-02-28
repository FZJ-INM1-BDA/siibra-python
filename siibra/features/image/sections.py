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

    def plot(self, *args, **kwargs):
        from ...locations import PointCloud, Plane
        from ...core.concept import get_registry
        from ...volumes import Volume
        import numpy as np
        import matplotlib.pyplot as plt

        query_vol = list(self.anchor._assignments.values())[0].query_structure
        assert isinstance(
            query_vol, Volume
        ), f"Cannot plot patch anchored at query structure of type {type(query_vol)}"
        plane = Plane.from_image(self)
        layermap = get_registry("Map").get("cortical layers bigbrain")
        layer_contours = {
            layername: plane.intersect_mesh(
                layermap.fetch(region=layername, format="mesh")
            )
            for layername in layermap.regions
        }
        crop_voi = self.section.intersection(query_vol.get_boundingbox())
        cropped_img = self.section.fetch(voi=crop_voi, resolution_mm=-1)
        phys2pix = np.linalg.inv(cropped_img.affine)

        # The probabilities can be assigned to the contour vertices with the
        # probability map.
        points = PointCloud(
            np.vstack(
                sum(
                    [
                        [s.coordinates for s in contour.crop(crop_voi)]
                        for contour in layer_contours["cortical layer 4 right"]
                    ],
                    [],
                )
            ),
            space="bigbrain",
        )
        probs = query_vol.evaluate_points(
            points
        )  # siibra warps points to MNI152 and reads corresponding PMAP values
        img_arr = cropped_img.get_fdata().squeeze().swapaxes(0, 1)

        fig = plt.figure()
        plt.imshow(img_arr, cmap="gray", origin="lower", vmin=0, vmax=2**16)
        X, _, Z = points.transform(phys2pix).coordinates.T
        plt.scatter(X, Z, s=2, c=probs, vmin=0)
        for p in self.profile:
            x, _, z = p.transform(phys2pix)
            plt.plot(x, z, "r.", ms=3)
        plt.axis("off")
        return fig
