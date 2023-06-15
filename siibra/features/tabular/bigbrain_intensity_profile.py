# Copyright 2018-2021
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

from . import spatial

from typing import Union, List
from ...locations import PointSet
from .. import anchor as _anchor
from .cortical_profile import BOUNDARIES
import numpy as np


class BigBrainIntensityProfile(
    spatial.PointCloud,
    category='cellular'
):

    DESCRIPTION = (
        "Cortical profiles of BigBrain staining intensities computed by Konrad Wagstyl, "
        "as described in the publication 'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678'."
        "Taken from the tutorial at https://github.com/kwagstyl/cortical_layers_tutorial "
        "and assigned to cytoarchitectonic regions of Julich-Brain."
    )

    def __init__(
        self,
        regionname: str,
        coords: Union[np.ndarray, List['tuple']],
        depths: list,
        values: list,
        boundary_depths: list,
    ):
        pointset = PointSet(coords, space="bigbrain")
        self._depths = depths
        self._values = values
        self.unit = "staining intensity"
        self.boundary_depths = [
            {b: vertex_depths[b[0]] for b in BOUNDARIES}
            for vertex_depths in boundary_depths
        ]
        anchor = _anchor.AnatomicalAnchor(
            species="Homo Sapiens",
            location=pointset.boundingbox,
            region=regionname
        )
        spatial.PointCloud.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Modified silver staining",
            anchor=anchor,
            pointset=pointset,
            value_headers=depths
        )

    def _check_sanity(self):
        # check plausibility of the profile
        assert isinstance(self._depths, (list, np.ndarray))
        assert isinstance(self._values, (list, np.ndarray))
        assert len(self._values) == len(self._depths)
        assert all(0 <= d <= 1 for d in self._depths)
        if self.boundaries_mapped:
            assert all(0 <= d <= 1 for d in self.boundary_depths.values())
            assert all(
                layerpair in BOUNDARIES
                for layerpair in self.boundary_depths.keys()
            )

    def assign_layer(self, coordinate, depth: float):
        """Compute the cortical layer for a given depth from the
        layer boundary positions. If no positions are available
        for this profile, return None."""
        assert 0 <= depth <= 1, "Cortical depth can only be in [0,1]"
        if not isinstance(coordinate, int):
            index = self.points.index(coordinate)
        if len(self.boundary_depths[index]) == 0:
            return None
        else:
            return max(
                [l2 for (l1, l2), d in self.boundary_depths[index].items() if d < depth]
            )
