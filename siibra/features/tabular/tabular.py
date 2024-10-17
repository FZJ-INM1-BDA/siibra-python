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
"""Base type of features in tabular formats."""

from zipfile import ZipFile
from .. import feature

from .. import anchor as _anchor

from ... import commons

import pandas as pd
from textwrap import wrap


class Tabular(feature.Feature):
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
        data: pd.DataFrame,  # sample x feature dimension
        datasets: list = [],
        prerelease: bool = False,
        id: str = None,
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
            prerelease=prerelease,
            id=id,
        )
        self._data_cached = data

    @property
    def data(self):
        return self._data_cached.copy()

    def _export(self, fh: ZipFile):
        super()._export(fh)
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
            except ImportError:
                commons.logger.error("matplotlib not available. Plotting of fingerprints disabled.")
                return None
            # default kwargs
            if kwargs.get("error_y") is None:
                kwargs["yerr"] = kwargs.get("yerr", 'std' if 'std' in self.data.columns else None)
                yerr_label = f" \u00b1 {kwargs.get('yerr')}" if kwargs.get('yerr') else ''
            kwargs["width"] = kwargs.get("width", 0.95)
            kwargs["ylabel"] = kwargs.get(
                "ylabel",
                f"{kwargs['y']}{yerr_label} {self.unit if hasattr(self, 'unit') else ''}"
            )
            kwargs["grid"] = kwargs.get("grid", True)
            kwargs["legend"] = kwargs.get("legend", False)
            ax = self.data.plot(*args, backend=backend, **kwargs)
            ax.set_title(ax.get_title(), fontsize="medium")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
            plt.tight_layout()
            return ax
        elif backend == "plotly":
            kwargs["title"] = kwargs["title"].replace("\n", "<br>")
            kwargs["error_y"] = kwargs.get("error_y", 'std' if 'std' in self.data.columns else None)
            error_y_label = f" &plusmn; {kwargs.get('error_y')}" if kwargs.get('error_y') else ''
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
