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

from . import tabular

from .. import anchor as _anchor
from ...locations import boundingbox as _boundingbox
from ...locations import PointSet
from ...retrieval.requests import MultiSourcedRequest

import pandas as pd
from typing import List


class Spatial(tabular.Tabular):
    def __init__(
        self,
        description: str,
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
        datasets: list = []
    ):
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            data=None,  # lazy loader below
            datasets=datasets
        )
        assert isinstance(self.anchor.location, _boundingbox.BoundingBox), \
            "spatial feautures must be anchored to a bounding box."

    @property
    def boundingbox(self):
        return self.anchor.location

    @property
    def space(self):
        return self.boundingbox.space


class PointCloud(Spatial):
    def __init__(
        self,
        description: str,
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
        pointset: None,
        loader: MultiSourcedRequest = None,
        value_headers: list = [],
        datasets: list = []
    ):
        Spatial.__init__(
            self,
            description=description,
            modality=modality,
            anchor=anchor,
            datasets=datasets
        )
        self._loader = loader
        self._pointset_cached = pointset
        self._value_headers = value_headers

    def _load(self):
        headers, coordinates, self._values = self._loader.data
        if len(self._value_headers):
            self._value_headers = headers
        self._pointset_cached = PointSet(coordinates, self.anchor.space)

    def as_pointset(self) -> PointSet:
        if self._pointset_cached is None:
            self._load()
        return self._pointset_cached

    @property
    def points(self) -> List[tuple]:
        return self.as_pointset().as_list()

    @property
    def data(self) -> pd.DataFrame:
        """Return a pandas DataFrame representing the profile."""
        # self._check_sanity()
        coords = pd.Series(self.points, name='coordinate')
        values = pd.DataFrame(self._values, columns=self._value_headers)
        return pd.concat([coords, values], axis=1)
