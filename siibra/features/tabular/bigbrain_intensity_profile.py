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

from typing import Union, List, Tuple
from ...locations import PointSet
from .. import anchor as _anchor
from . import cortical_profile
import numpy as np


class BigBrainIntensityProfile(
    spatial.PointCloud,
    cortical_profile.CorticalProfile,
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
        boundary_positions: list,
    ):
        pointset = PointSet(coords, space="bigbrain")
        modality = "Modified silver staining"
        self.boundary_depths = [
            {b: vertex_depths[b[0]] for b in self.BOUNDARIES}
            for vertex_depths in boundary_positions
        ]
        anchor = _anchor.AnatomicalAnchor(
            species="Homo Sapiens",
            location=pointset.boundingbox,
            region=regionname
        )
        spatial.PointCloud.__init__(
            self,
            description=self.DESCRIPTION,
            modality=modality,
            anchor=anchor,
            pointset=pointset,
            value_headers=depths
        )
        cortical_profile.CorticalProfile.__init__(
            self,
            description=self.DESCRIPTION,
            modality=modality,
            anchor=anchor,
            depths=depths,
            values=values,
            unit="staining intensity",
            boundary_positions=[
                {b: vertex_depths[b[0]] for b in self.BOUNDARIES}
                for vertex_depths in boundary_positions
            ]
        )

    def plot(self, coordinate_or_index: Union[int, Tuple], **kwargs):
        if isinstance(coordinate_or_index, tuple):
            try:
                index = self.points.index(coordinate_or_index)
            except Exception:
                raise ValueError(
                    f"The coordinate {coordinate_or_index} cannot be found"
                    " within the coordinates of the profile."
                )
        else:
            index = coordinate_or_index
        return self.data.iloc[index, 1:].plot(**kwargs)
