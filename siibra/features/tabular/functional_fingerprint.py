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

from typing import List, TYPE_CHECKING, Callable

from ..feature import Compoundable
from .tabular import Tabular
from ...retrieval import HttpRequest

if TYPE_CHECKING:
    from ..anchor import AnatomicalAnchor


class FunctionalFingerprint(
    Tabular,
    Compoundable,
    category="functional",
    configuration_folder="features/tabular/fingerprints/functional",
):

    _filter_attrs = ["modality", "parcellation", "region"]
    _compound_attrs = ["modality", "parcellation"]

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        file: str,
        region: str,
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
        self._region = region

    @property
    def parcellation(self):
        assert len(self.anchor.parcellations) == 1
        return self.anchor.parcellations[0]

    @property
    def region(self):
        return self.parcellation.get_region(self._region)

    @property
    def data(self):
        if self._data_cached is None:
            self._data_cached = self._loader.data
        return self._data_cached.copy()

    @classmethod
    def _merge_elements(
        cls,
        elements: List["FunctionalFingerprint"],
        description: str,
        modality: str,
        anchor: "AnatomicalAnchor",
    ):
        assert len({f.cohort for f in elements}) == 1
        # merged = cls(
        # )

    def plot(self, region: str, **kwargs):
        self.data.plot(
            x=region,
            backend="plotly",
            color_continuous_scale="BlueRed",
            kind="barh",
            labels={"labels": "contrast"},
            **kwargs,
        )

    # def plot_results(
    #     table: pd.DataFrame, mp: siibra._parcellationmap.Map, task: str, contrast: str
    # ):
    #     from nilearn import plotting

    #     table.set_index("labels", inplace=True)
    #     task_table = table[table["task"] == task].drop(columns=["task"]).T

    #     plotting.view_img(
    #         mp.colorize(task_table[contrast].to_dict()).fetch(),
    #         symmetric_cmap=False,
    #         cmap="magma",
    #     ).open_in_browser()
