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

from . import feature

from .. import anchor as _anchor

from ... import commons

import pandas as pd
from textwrap import wrap


class Tabular(feature.Feature):
    """
    Represents a table of different measures anchored to a brain location.

    Columns represent different types of values, while rows represent
    different samples. The number of columns might thus be intrepreted
    as the feature dimension.

    As an example, receptor fingerprints use rows
    to represent different neurotransmitter receptors, and separate
    columns for the mean and standard deviations measure across multiple
    tissue samples.
    """

    def __init__(
        self,
        description: str,
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
        data: pd.DataFrame,  # sample x feature dimension
        datasets: list = []
    ):
        feature.Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets
        )
        self._data_cached = data

    @property
    def data(self):
        return self._data_cached

    def plot(self, **kwargs):
        """ Create a bar plot of a columns of the data."""

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            commons.logger.error("matplotlib not available. Plotting of fingerprints disabled.")
            return None

        wrapwidth = kwargs.pop("textwrap") if "textwrap" in kwargs else 40

        # default kwargs
        kwargs["y"] = kwargs.get("y", self.data.columns[0])
        kwargs["yerr"] = kwargs.get("yerr", 'std' if 'std' in self.data.columns else None)
        kwargs["width"] = kwargs.get("width", 0.95)
        kwargs["ylabel"] = kwargs.get(
            "ylabel",
            f"{kwargs['y']} {self.unit if hasattr(self, 'unit') else ''}"
        )
        kwargs["xlabel"] = kwargs.get("xlabel")
        kwargs["title"] = kwargs.get(
            "title",
            "\n".join(wrap(f"{self.modality} in {', '.join({_.name for _ in self.anchor.regions})}", wrapwidth))
        )
        kwargs["grid"] = kwargs.get("grid", True)
        kwargs["legend"] = kwargs.get("legend", False)
        ax = self.data.plot(kind="bar", **kwargs)
        ax.set_title(ax.get_title(), fontsize="medium")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
        plt.tight_layout()
