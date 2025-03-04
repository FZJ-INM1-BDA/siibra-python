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

from . import cortical_profile
from . import tabular, cell_reader, layer_reader
from .. import anchor as _anchor
from ... import commons
from ...retrieval import requests

import pandas as pd
import numpy as np


class LayerwiseCellDensity(
    tabular.Tabular,
    configuration_folder="features/tabular/layerstatistics/celldensity",
    category='cellular'
):

    DESCRIPTION = (
        "Layerwise estimated densities of detected cell bodies  (in detected cells per 0.1 cube millimeter) "
        "obtained by applying a Deep Learning based instance segmentation algorithm (Contour Proposal Network; Upschulte "
        "et al., Neuroimage 2022) to a 1 micron resolution cortical image patch prepared with modified Silver staining. "
        "Densities have been computed per cortical layer after manual layer segmentation, by dividing the number of "
        "detected cells in that layer with the area covered by the layer. Therefore, each profile contains 6 measurement points. "
        "The cortical depth is estimated from the measured layer thicknesses."
    )

    def __init__(
        self,
        segmentfiles: list,
        layerfiles: list,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = [],
        id: str = None,
        prerelease: bool = False,
    ):
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Cell body density",
            anchor=anchor,
            datasets=datasets,
            data=None,  # lazy loading below
            id=id,
            prerelease=prerelease,
        )
        self.unit = "# detected cells/0.1mm3"
        self._filepairs = list(zip(segmentfiles, layerfiles))
        self._densities = None

    def _load_densities(self):
        density_dict = {}
        for i, (cellfile, layerfile) in enumerate(self._filepairs):
            try:
                cells = requests.HttpRequest(cellfile, func=cell_reader).data
                layers = requests.HttpRequest(layerfile, func=layer_reader).data
            except requests.SiibraHttpRequestError as e:
                print(str(e))
                commons.logger.error(f"Skipping to bootstrap a {self.__class__.__name__} feature, cannot access file resource.")
                continue
            counts = cells.layer.value_counts()
            areas = layers["Area(micron**2)"]
            indices = np.intersect1d(areas.index, counts.index)
            density_dict[i] = counts[indices] / areas * 100 ** 2 * 5
        return pd.DataFrame(density_dict)

    @property
    def data(self):
        if self._data_cached is None:
            densities = self._load_densities()
            self._data_cached = pd.DataFrame(
                np.array([
                    list(densities.mean(axis=1)),
                    list(densities.std(axis=1))
                ]).T,
                columns=['mean', 'std'],
                index=[cortical_profile.CorticalProfile.LAYERS[_] for _ in densities.index]
            )
            self._data_cached.index.name = 'layer'
        return self._data_cached
