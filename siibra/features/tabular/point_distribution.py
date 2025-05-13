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

from ...retrieval.requests import HttpRequest
from ...locations import BoundingBox, PointSet
from ...core.space import Space
from ...volumes.volume import from_pointset

from typing import Dict, Callable, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from ...volumes.volume import Volume


class PointDistribution(
    tabular.Tabular,
    configuration_folder="features/tabular/point_distribution",
    category='cellular'
):

    def __init__(
        self,
        modality: str,
        space_spec: dict,
        file: Dict[str, str],
        description: str = "",
        decoder: Callable = None,
        datasets: list = []
    ):
        space = Space.get_instance(space_spec.get('@id') or space_spec.get('name'))
        anchor = _anchor.AnatomicalAnchor(
            species=space.species,
            location=space.get_template().boundingbox,
            region=None
        )
        tabular.Tabular.__init__(
            self,
            description=description,
            modality=modality,
            anchor=anchor,
            data=None,  # lazy loading below
            datasets=datasets,
        )
        assert len(file) == 1
        self._index, filename = list(*file.items())
        self._loader = HttpRequest(filename, decoder)
        self._pointset_cached = None

    @property
    def index(self):
        return self._index

    def __len__(self):
        """Total number of coordinates."""
        return len(self.as_pointset())

    def as_pointset(self) -> PointSet:
        coordinates, *self._values = self.data
        return PointSet(coordinates, self.anchor.space)

    @property
    def boundingbox(self) -> BoundingBox:
        return self.as_pointset().boundingbox

    @property
    def data(self) -> pd.DataFrame:
        """Return a pandas DataFrame representing the profile."""
        if self._data_cached is None:
            self._data_cached = self._loader.get()
        return self._data_cached.copy()

    def plot(self, *args, backend='matplotlib', **kwargs):
        if self.data.shape[0] <= 3:
            raise NotImplementedError(
                "The point distribution does not contain any value data."
            )
        kind = kwargs.pop('kind', "hist")
        return self.data.iloc[:, 3:].plot(
            *args, backend=backend, kind=kind, **kwargs
        )

    def get_kde_volume(
        self, normalize: bool = True, min_num_point: int = 0, **kwargs
    ) -> "Volume":
        """
        Get the kernel density estimate from the points using their average
        uncertainty on the reference space template the coordinates belongs to.

        Parameters
        ----------
        normalize : bool, optional
        min_num_points : int, default 0

        Returns
        -------
        Volume
            _description_
        """
        return from_pointset(
            points=self.as_pointset(),
            target=self.anchor.space.get_template(**kwargs),
            normalize=normalize,
            min_num_point=min_num_point
        )
