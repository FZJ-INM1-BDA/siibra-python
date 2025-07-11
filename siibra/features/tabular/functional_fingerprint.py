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

from typing import TYPE_CHECKING, Callable

from .tabular import Tabular
from ...retrieval import HttpRequest

if TYPE_CHECKING:
    from ..anchor import AnatomicalAnchor


class FunctionalFingerprint(
    Tabular,
    category="functional",
    configuration_folder="features/tabular/fingerprints/functional",
):

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        file: str,
        decoder: Callable = None,
        datasets: list = [],
        id: str = None,
        prerelease: bool = False,
    ):
        Tabular.__init__(
            self,
            modality="functional fingerprint",
            anchor=anchor,
            datasets=datasets,
            data=None,  # lazy loading below
            id=id,
            prerelease=prerelease,
            description=None,
        )
        self._loader = HttpRequest(file, func=decoder)

    @property
    def name(self):
        return f"{super().name}: {self.anchor._regionspec} ({self.anchor._parcellation_version})"

    @property
    def data(self):
        if self._data_cached is None:
            self._data_cached = self._loader.data
        return self._data_cached.copy()

    def plot(self, *args, backend="matplotlib", **kwargs):
        if backend == "matplotlib":
            return super().plot(*args, backend=backend, **kwargs)
        elif backend == "plotly":
            df_2_plot = self.data.reset_index()  # plotly backend does not support multiindex atm
            df_2_plot["labels/task"] = df_2_plot["labels"] + "/" + df_2_plot["task"]
            return df_2_plot.plot(
                x=self.anchor._regionspec,
                y="labels/task",
                color="task",
                backend="plotly",
                color_continuous_scale="jet",
                kind="barh",
                labels={"labels": "contrast"},
                width=800,
                height=3000,
                **kwargs,
            )
        else:
            return self.data.plot(*args, backend=backend, **kwargs)
