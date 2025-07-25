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
"""Base type of features in tabular formats."""

from zipfile import ZipFile
from typing import Callable

import pandas as pd
from textwrap import wrap

from .. import feature
from .. import anchor as _anchor
from ...commons import logger
from ...retrieval import requests


class Tabular(feature.Feature, category="generic", configuration_folder="features/tabular"):
    """
    Represents a table of different measures anchored to a brain location.

    Columns represent different types of values, while rows represent different
    samples. The number of columns might thus be interpreted as the feature
    dimension.

    As an example, receptor fingerprints use rows to represent different
    neurotransmitter receptors, and separate columns for the mean and standard
    deviations measure across multiple tissue samples.
    """

    def __init__(
        self,
        description: str,
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
        file: str = None,
        decoder: Callable = None,
        data: pd.DataFrame = None,  # sample x feature dimension
        datasets: list = [],
        id: str = None,
        prerelease: bool = False,
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
            id=id,
            prerelease=prerelease
        )
        self._loader = None if file is None else requests.HttpRequest(file, func=decoder)
        if file is not None:
            assert data is None
        self._data_cached = data

    @property
    def data(self):
        if self._loader is not None:
            self._data_cached = self._loader.get()
        return self._data_cached.copy()

    def _to_zip(self, fh: ZipFile):
        super()._to_zip(fh)
        fh.writestr("tabular.csv", self.data.to_csv())

    def plot(self, *args, backend="matplotlib", **kwargs):
        """
        Create a bar plot of a columns of the data.
        Parameters
        ----------
        backend: str
            "matplotlib", "plotly", or others supported by pandas DataFrame
            plotting backend.
        **kwargs
            takes Matplotlib.pyplot keyword arguments
        """
        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40
        kwargs["title"] = kwargs.get(
            "title",
            "\n".join(wrap(
                f"{self.modality} in {', '.join({_.name for _ in self.anchor.regions})}",
                wrapwidth
            ))
        )
        kwargs["kind"] = kwargs.get("kind", "bar")
        kwargs["y"] = kwargs.get("y", self.data.columns[0])
        if backend == "matplotlib":
            try:
                import matplotlib.pyplot as plt
            except ImportError as e:
                logger.error(
                    "matplotlib not available. Please install matplotlib or use or another backend such as plotly."
                )
                raise e
            # default kwargs
            if kwargs.get("error_y") is None:
                kwargs["yerr"] = kwargs.get("yerr", 'std' if 'std' in self.data.columns else None)
                yerr_label = f" \u00b1 {kwargs.get('yerr')}" if kwargs.get('yerr') else ''
            if kwargs.get('kind') == 'bar':
                kwargs["width"] = kwargs.get("width", 0.8)
                kwargs["edgecolor"] = kwargs.get('edgecolor', 'black')
                kwargs["linewidth"] = kwargs.get('linewidth', 1.0)
                kwargs["capsize"] = kwargs.get('capsize', 4)
            kwargs["ylabel"] = kwargs.get(
                "ylabel",
                f"{kwargs['y']}{yerr_label}" + f"\n{self.unit}" if hasattr(self, 'unit') else ""
            )
            kwargs["grid"] = kwargs.get("grid", True)
            kwargs["legend"] = kwargs.get("legend", False)
            kwargs["color"] = kwargs.get('color', 'darkgrey')

            xticklabel_rotation = kwargs.get("rot", 60)
            ax = self.data.plot(*args, backend=backend, **kwargs)
            ax.set_title(ax.get_title(), fontsize="medium")
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=xticklabel_rotation,
                ha='center' if xticklabel_rotation % 90 == 0 else 'right'
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            return ax
        elif backend == "plotly":
            kwargs["title"] = kwargs["title"].replace("\n", "<br>")
            kwargs["error_y"] = kwargs.get("error_y", 'std' if 'std' in self.data.columns else None)
            error_y_label = f" &plusmn; {kwargs.get('error_y')}<br>" if kwargs.get('error_y') else ''
            kwargs["labels"] = {
                "index": kwargs.pop("xlabel", None) or kwargs.pop("index", ""),
                "value": kwargs.pop("ylabel", None) or kwargs.pop(
                    "value",
                    f"{kwargs.get('y')}{error_y_label} {self.unit if hasattr(self, 'unit') else ''}"
                )
            }
            fig = self.data.plot(*args, backend=backend, **kwargs)
            fig.update_layout(
                yaxis_title=kwargs["labels"]['value'],
                title=dict(
                    automargin=True, yref="container", xref="container",
                    pad=dict(t=40), xanchor="left", yanchor="top"
                )
            )
            return fig
        else:
            return self.data.plot(*args, backend=backend, **kwargs)
