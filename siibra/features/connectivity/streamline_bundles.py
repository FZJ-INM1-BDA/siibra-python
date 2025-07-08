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

from typing import Callable, TypedDict, Dict, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..feature import Feature
from .. import anchor as _anchor
from ...locations import Contour

if TYPE_CHECKING:
    from ...retrieval.repositories import RepositoryConnector


class _Transform(TypedDict):
    affine: np.array
    space: dict


class StreamlineFiberBundle(
    Feature,
    configuration_folder="features/connectivity/streamlinefiberbundles",
    category="connectivity",
):

    def __init__(
        self,
        modality: str,
        connector: "RepositoryConnector",
        decode_func: Callable,
        filename: str,
        bundle_id: str,
        anchor: _anchor.AnatomicalAnchor,
        transform: _Transform,
        description: str = "",
        datasets: list = [],
        cohort: str = None,
        id: str = None,
        prerelease: bool = False,
    ):
        Feature.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
            id=id,
            prerelease=prerelease,
        )
        self._bundle_id = bundle_id
        self.cohort = cohort.upper() if isinstance(cohort, str) else cohort
        self._connector = connector
        self._filename = filename
        self._decode_func = decode_func
        self.transform = transform

    @property
    def name(self):
        return f"{self.bundle_id} - {super().name}"

    def __len__(self):
        return len(self.data.index.unique())

    def get_fibers(self) -> Dict[str, Contour]:
        fiber_ids = self.data.index.unique()
        return {
            fiber_id: Contour(self.data.loc[fiber_id], space=None).transform(
                self.transform["affine"], space=self.transform["space"]
            )  # TODO: remove the need for transformation
            for fiber_id in fiber_ids
        }

    @property
    def data(self) -> pd.DataFrame:
        return self._connector.get(self._filename, decode_func=self._decode_func)

    def plot(self, *args, backend="nilearn", **kwargs):
        from nilearn import plotting

        coords = np.vstack([c.coordinates for c in self.fibers.values()])
        return plotting.plot_markers(
            self.data.index.tolist(),
            coords,
            node_size=3,
        )
