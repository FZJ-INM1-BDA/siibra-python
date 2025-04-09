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

import numpy as np
import pandas as pd
from textwrap import wrap

from . import tabular, cell_reader, layer_reader
from .. import anchor as _anchor
from ... import commons
from ...retrieval import requests


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
    BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR = 1.931

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
        self.unit = "# detected cells / $0.1mm^3$"
        self._filepairs = list(zip(segmentfiles, layerfiles))
        self._densities = None

    def _load_densities(self):
        data = []
        for cellfile, layerfile in self._filepairs:
            try:
                cells = requests.HttpRequest(cellfile, func=cell_reader).data
                layers = requests.HttpRequest(layerfile, func=layer_reader).data
            except requests.SiibraHttpRequestError as e:
                print(str(e))
                commons.logger.error(f"Skipping to bootstrap a {self.__class__.__name__} feature, cannot access file resource.")
                continue
            counts = cells.layer.value_counts()
            # compute the volumetric shrinkage corrections in the same ways as it was used
            # for the pdf reports in the underlying dataset
            shrinkage_volumetric = self.BIGBRAIN_VOLUMETRIC_SHRINKAGE_FACTOR
            layer_volumes = (
                layers["Area(micron**2)"]  # this is the number of pixels, shrinkage corrected from the dataset
                * 20  # go to cube micrometer in one patch with 20 micron thickness
                * np.cbrt(shrinkage_volumetric)  # compensate linear shrinkage for 3rd dimension
                / 100 ** 3  # go to 0.1 cube millimeter
            )
            fields = cellfile.split("/")
            for layer in layer_volumes.index:
                data.append({
                    'layer': layer,
                    'layername': layers["Name"].loc[layer],
                    'counts': counts.loc[layer],
                    'area_mu2': layers["Area(micron**2)"].loc[layer],
                    'volume': layer_volumes.loc[layer],
                    'density': counts.loc[layer] / layer_volumes.loc[layer],
                    'regionspec': fields[-5],
                    'section': int(fields[-3]),
                    'patch': int(fields[-2]),
                })
        return pd.DataFrame(data)

    @property
    def data(self):
        if self._data_cached is None:
            self._data_cached = self._load_densities()
            # self._data_cached.index.name = 'layer'
        return self._data_cached

    def plot(self, *args, backend="matplotlib", **kwargs):
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40
        kwargs["title"] = kwargs.pop(
            "title",
            "\n".join(wrap(
                f"{self.modality} in {self.anchor._regionspec or self.anchor.location}",
                wrapwidth
            ))
        )
        kwargs["kind"] = kwargs.get("kind", "box")
        kwargs["ylabel"] = kwargs.get(
            "ylabel",
            f"\n{self.unit}" if hasattr(self, 'unit') else ""
        )
        if backend == "matplotlib":
            if kwargs["kind"] == "box":
                from matplotlib.pyplot import tight_layout

                title = kwargs.pop("title")
                default_kwargs = {
                    "grid": True,
                    'by': "layername",
                    'column': ['density'],
                    'showfliers': False,
                    'xlabel': 'layer',
                    'color': 'dimgray',
                }
                ax, *_ = self.data.plot(*args, backend=backend, **{**default_kwargs, **kwargs})
                for i, (layer, d) in enumerate(self.data.groupby('layername')):
                    ax.scatter(
                        np.random.normal(i + 1, 0.05, len(d.density)),
                        d.density,
                        c='b', s=3
                    )
                ax.set_title(title)
                tight_layout()
                return ax
            return self.data.plot(*args, backend=backend, **kwargs)
        elif backend == "plotly":
            kwargs["title"] = kwargs["title"].replace('\n', "<br>")
            return self.data.plot(y='density', x='layer', backend=backend, **kwargs)
        else:
            return self.data.plot(*args, backend=backend, **kwargs)
