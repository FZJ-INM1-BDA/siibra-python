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

from .. import anchor as _anchor
from ..basetypes import cortical_profile, tabular

from ... import commons
from ...retrieval import requests

import pandas as pd
import numpy as np
from io import BytesIO


class LayerwiseCellDensity(tabular.Tabular, configuration_folder="features/fingerprints/celldensity"):

    DESCRIPTION = (
        "Layerwise estimated densities of detected cell bodies  (in detected cells per 0.1 cube millimeter) "
        "obtained by applying a Deep Learning based instance segmentation algorithm (Contour Proposal Network; Upschulte "
        "et al., Neuroimage 2022) to a 1 micron resolution cortical image patch prepared with modified Silver staining. "
        "Densities have been computed per cortical layer after manual layer segmentation, by dividing the number of "
        "detected cells in that layer with the area covered by the layer. Therefore, each profile contains 6 measurement points. "
        "The cortical depth is estimated from the measured layer thicknesses."
    )

    @classmethod
    def CELL_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0).astype(
            {"layer": int, "label": int}
        )

    @classmethod
    def LAYER_READER(cls, b):
        return pd.read_csv(BytesIO(b[2:]), delimiter=" ", header=0, index_col=0)

    def __init__(
        self,
        segmentfiles: list,
        layerfiles: list,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = [],
    ):
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Cell body density",
            anchor=anchor,
            datasets=datasets,
            data=None  # lazy loading below
        )
        self.unit = "# detected cells/0.1mm3"
        self._filepairs = list(zip(segmentfiles, layerfiles))
        self._densities = None

    def _load_densities(self):
        density_dict = {}
        for i, (cellfile, layerfile) in enumerate(self._filepairs):
            try:
                cells = requests.HttpRequest(cellfile, func=self.CELL_READER).data
                layers = requests.HttpRequest(layerfile, func=self.LAYER_READER).data
            except requests.SiibraHttpRequestError as e:
                print(str(e))
                commons.logger.error(f"Skipping to bootstrap a {self.__class__.__name__} feature, cannot access file resource.")
                continue
            counts = cells.layer.value_counts()
            areas = layers["Area(micron**2)"]
            density_dict[i] = counts[areas.index] / areas * 100 ** 2 * 5
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

    @property
    def key(self):
        assert len(self.species) == 1
        return commons.create_key("{}_{}_{}".format(
            self.dataset_id,
            self.species[0]['name'],
            self.regionspec
        ))

    def plot(self, **kwargs):
        kwargs['ylabel'] = self.unit
        super().plot(**kwargs)
