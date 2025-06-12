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
        assert len({e.parcellation for e in elements}) == 1
        assert len({"/".join(elements[0]._loader.url.split("/")[:-1]) for e in elements}) == 1
        merged = cls(
            anchor=anchor,
            file="/".join(elements[0]._loader.url.split("/")[:-1])
            + "/functional_profile.csv",
            region=elements[0].parcellation,
            decoder=elements[0]._loader.func,
        )
        return merged

    def plot(self, *args, backend="matplotlib", **kwargs):
        if backend == "matplotlib":
            return super().plot(*args, backend=backend, **kwargs)
        elif backend == "plotly":
            df_2_plot = self.data.reset_index()
            df_2_plot["labels/task"] = df_2_plot["labels"] + "/" + df_2_plot["task"]
            return df_2_plot.plot(
                x=self.region.name,
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
