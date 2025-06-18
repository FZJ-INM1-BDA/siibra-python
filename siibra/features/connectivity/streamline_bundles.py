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

from typing import Callable, List, TypedDict, Dict, TYPE_CHECKING
from hashlib import md5

import numpy as np
import pandas as pd

from ..feature import Feature, Compoundable
from .. import anchor as _anchor
from ...locations import Contour

if TYPE_CHECKING:
    from ...retrieval.repositories import RepositoryConnector


class _Transform(TypedDict):
    affine: np.array
    space: dict


class StreamlineFiberBundle(
    Feature,
    Compoundable,
    configuration_folder="features/connectivity/streamlinefiberbundles",
    category="connectivity",
):
    _filter_attrs = ["modality", "cohort", "subject", "bundle_id"]
    _compound_attrs = ["modality", "cohort"]

    def __init__(
        self,
        modality: str,
        regions: List[str],
        connector: "RepositoryConnector",
        decode_func: Callable,
        filename: str,
        bundle_id: str,
        anchor: _anchor.AnatomicalAnchor,
        transform: _Transform,
        description: str = "",
        datasets: list = [],
        cohort: str = None,
        subject: str = None,
        feature: str = None,
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
        self._regions = regions
        self._filename = filename
        self._decode_func = decode_func
        self._subject = subject
        self.feature = feature
        self.transform = transform
        self._fibers: List[Contour] = None

    @property
    def bundle_id(self):
        return self._bundle_id

    @property
    def id(self):
        return super().id + "--" + md5(self.bundle_id.encode("utf-8")).hexdigest()

    @property
    def fibers(self) -> Dict[str, Contour]:
        if self._fibers is None:
            assert isinstance(self.data, pd.DataFrame)
            self._fibers = dict()
            for fiber_id in self.data.index.unique():
                fiber = Contour(self.data.loc[fiber_id], space=self.transform["space"])
                if self.transform["affine"] is not None:
                    fiber = fiber.transform(
                        self.transform["affine"], space=self.transform["space"]
                    )
                self._fibers[fiber_id] = fiber
        return self._fibers

    @property
    def data(self) -> pd.DataFrame:
        return self._connector.get(self._filename, decode_func=self._decode_func)

    @property
    def regions(self) -> List[str]:
        if isinstance(self._regions, str):
            regions_df = self._connector.get(self._regions, decode_func=self._decode_func)
            self._regions = (
                regions_df.loc[self.bundle_id].loc[lambda s: s.eq(1)].index.tolist()
            )
        return self._regions

    @property
    def subject(self):
        return self._subject

    def _merge_elements(self):
        pass

    def plot(self, *args, backend="nilearn", **kwargs):
        from nilearn import plotting

        coords = np.vstack([c.coordinates for c in self.fibers.values()])
        return plotting.plot_markers(
            self.data.index.tolist(),
            coords,
            node_size=3,
        )
