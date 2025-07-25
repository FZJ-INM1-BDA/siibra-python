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

from textwrap import wrap
from typing import List

import numpy as np
import pandas as pd

from . import tabular
from .. import anchor as _anchor
from ...commons import logger
from ...vocabularies import RECEPTOR_SYMBOLS


class ReceptorDensityFingerprint(
    tabular.Tabular,
    configuration_folder="features/tabular/fingerprints/receptor",
    category='molecular'
):

    DESCRIPTION = (
        "Fingerprint of densities (in fmol/mg protein) of receptors for classical neurotransmitters "
        "obtained by means of quantitative in vitro autoradiography. The fingerprint provides average "
        "density measurements for different receptors measured in tissue samples from different subjects "
        "together with the corresponding standard deviations. "
    )

    def __init__(
        self,
        tsvfile: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = [],
        id: str = None,
        prerelease: bool = False,
    ):
        """ Generate a receptor fingerprint from a URL to a .tsv file
        formatted according to the structure used by Palomero-Gallagher et al.
        """
        tabular.Tabular.__init__(
            self,
            description=self.DESCRIPTION,
            modality="Neurotransmitter receptor density",
            anchor=anchor,
            file=tsvfile,
            data=None,  # lazy loading below
            datasets=datasets,
            id=id,
            prerelease=prerelease,
        )

    @property
    def unit(self) -> str:
        return self._loader.data.iloc[:, -1][0]

    @property
    def receptors(self) -> List[str]:
        return list(self.data.index)

    @property
    def neurotransmitters(self) -> List[str]:
        # TODO quite a lot of receptor features have undecipherable symbols, mainly double quoted receptor
        # Likely ill-formed tsv's
        return [
            "{} ({})".format(
                RECEPTOR_SYMBOLS[t]['neurotransmitter']['label'],
                RECEPTOR_SYMBOLS[t]['neurotransmitter']['name'],
            ) if t in RECEPTOR_SYMBOLS else
            f"{t} (undeciphered)"
            for t in self.receptors
        ]

    @property
    def data(self):
        if self._data_cached is None:
            label_col, mean_col, std_col = list(self._loader.data.columns)[:3]
            self._data_cached = pd.DataFrame(
                np.array([
                    self._loader.data[mean_col],
                    self._loader.data[std_col]
                ]).T,
                index=self._loader.data[label_col],
                columns=['mean', 'std']
            )
            self._data_cached.index.name = 'receptor'
        return self._data_cached.copy()

    @classmethod
    def parse_tsv_data(cls, data: dict):
        units = {list(v.values())[3] for v in data.values()}
        labels = list(data.keys())
        assert len(units) == 1
        try:
            mean = [data[_]["density (mean)"] for _ in labels]
            std = [data[_]["density (sd)"] for _ in labels]
        except KeyError as e:
            print(str(e))
            logger.error("Could not parse fingerprint from this dictionary")
        return {
            'unit': next(iter(units)),
            'labels': labels,
            'means': [float(m) if m.isnumeric() else 0 for m in mean],
            'stds': [float(s) if s.isnumeric() else 0 for s in std],
        }

    def polar_plot(self, *args, backend='matplotlib', **kwargs):
        """
        Create a polar plot of the fingerprint.
        backend: str
            "matplotlib" or "plotly"
        """
        if backend == "matplotlib":
            try:
                import matplotlib.pyplot as plt
            except ImportError as e:
                logger.error(
                    "matplotlib not available. Please install matplotlib or use or another backend such as plotly."
                )
                raise e
            from collections import deque

            # default args
            wrapwidth = 40
            y = kwargs.pop("y") if "y" in kwargs else self.data.columns[0]
            yerr = kwargs.pop("yerr") if "yerr" in kwargs else None
            if yerr is None:
                yerr = 'std' if 'std' in self.data.columns else None
            ax = kwargs.pop("ax") if "ax" in kwargs else plt.subplot(111, projection="polar")

            datafield = y or self.data.columns[0]
            if yerr is None and 'std' in self.data.columns:
                yerr = 'std'
            # values = list(self.data[datafield])
            angles = deque(np.linspace(0, 2 * np.pi, self.data.shape[0] + 1)[:-1][::-1])
            angles.rotate(5)
            angles = list(angles)
            # for the values, repeat the first element to have a closed plot
            indices = list(range(self.data.shape[0])) + [0]
            y = list(self.data[datafield].iloc[indices])
            plt.plot(angles + [angles[0]], y, "k-", lw=3, **kwargs)
            if yerr:
                bounds0 = y - self.data[yerr].iloc[indices]
                plt.plot(angles + [angles[0]], bounds0, "k", lw=0.5, **kwargs)
                bounds1 = y + self.data[yerr].iloc[indices]
                plt.plot(angles + [angles[0]], bounds1, "k", lw=0.5, **kwargs)
            ax.set_xticks(angles)
            ax.set_xticklabels([_ for _ in self.data.index])
            ax.set_ylabel(self.unit)
            ax.set_title(
                "\n".join(wrap(f"{self.modality} anchored at {self.anchor._regionspec}", wrapwidth))
            )
            ax.tick_params(pad=9, labelsize=10)
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout()
            return ax
        elif backend == "plotly":
            from plotly.express import line_polar
            df = pd.DataFrame(
                {
                    "values": pd.concat(
                        [
                            self.data["mean"],
                            self.data["mean"] - self.data["std"],
                            self.data["mean"] + self.data["std"]
                        ]
                    ),
                    "cat": (
                        len(self.data) * ["mean"]
                        + len(self.data) * ["mean - std"]
                        + len(self.data) * ["mean + std"]
                    )
                }
            )
            return line_polar(
                df, r="values", theta=df.index, color="cat", line_close=True, **kwargs
            )
        else:
            raise NotImplementedError

    def plot(
        self,
        *args,
        receptors: List[str] = None,
        backend: str = "matplotlib",
        **kwargs
    ):
        """
        Create a bar plot of receptor density fingerprint.

        Parameters
        ----------
        receptors : List[str], optional
            Plot a subset of receptors.
        backend: str
            "matplotlib", "plotly", or others supported by pandas DataFrame
            plotting backend.
        **kwargs
            takes Matplotlib.pyplot keyword arguments
        """
        kwargs['xlabel'] = ""
        kwargs["backend"] = backend
        og_data = self.data
        if receptors is not None:
            self._data_cached = og_data.loc[receptors]
        fig = super().plot(*args, **kwargs)
        self._data_cached = og_data
        return fig
