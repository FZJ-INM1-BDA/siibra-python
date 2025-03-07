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

from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

from . import tabular
from . import cortical_profile

if TYPE_CHECKING:
    from ...features.anchor import AnatomicalAnchor


class LayerwiseBigBrainIntensities(
    tabular.Tabular,
    category='cellular'
):

    DESCRIPTION = (
        "Layerwise averages and standard deviations of of BigBrain staining intensities "
        "computed by Konrad Wagstyl, as described in the publication "
        "'Wagstyl, K., et al (2020). BigBrain 3D atlas of "
        "cortical layers: Cortical and laminar thickness gradients diverge in sensory and "
        "motor cortices. PLoS Biology, 18(4), e3000678. "
        "http://dx.doi.org/10.1371/journal.pbio.3000678."
        "The data is taken from the tutorial at "
        "https://github.com/kwagstyl/cortical_layers_tutorial. Each vertex is "
        "assigned to the regional map when queried."
    )

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        means: list,
        stds: list,
    ):
        data = pd.DataFrame(
            np.array([means, stds]).T,
            columns=['mean', 'std'],
            index=list(cortical_profile.CorticalProfile.LAYERS.values())[1: -1]
        )
        data.index.name = "layer"
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Modified silver staining",
            anchor=anchor,
            data=data
        )
