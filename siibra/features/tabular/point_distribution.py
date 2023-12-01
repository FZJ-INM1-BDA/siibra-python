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
from ..feature import Compoundable
from .. import anchor as _anchor

from ...retrieval.requests import HttpRequest
from ...locations import PointSet
from ...core.space import Space
from ...volumes.volume import from_pointset

from typing import Callable, TYPE_CHECKING, Union, List

if TYPE_CHECKING:
    from ...volumes.volume import Volume
    from ...locations import BoundingBox
    from pandas import DataFrame


class PointDistribution(
    tabular.Tabular,
    Compoundable,
    configuration_folder="features/tabular/point_distribution",
    category='cellular'
):
    """
    Represents a data frame with at least 3 columns (x, y, z, value#0, value#1,...)
    where the coordinates corresponds to a reference space,
    """

    _filter_attrs = ["modality", "subject"]
    _compound_attrs = ["modality"]

    def __init__(
        self,
        modality: str,
        space_spec: dict,
        subject: str,
        filename: str,
        description: str = "",
        decoder: Callable = None,
        datasets: list = []
    ):
        space = Space.get_instance(space_spec.get('@id') or space_spec.get('name'))
        anchor = _anchor.AnatomicalAnchor(
            species=space.species,
            location=space.get_template().get_boundingbox(clip=False),
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
        self._loader = HttpRequest(filename, decoder)
        self._subject = subject

    @property
    def subject(self):
        return self._subject

    def __len__(self):
        """Total number of coordinates."""
        return len(self.as_pointset())

    def as_pointset(
        self,
        sigma_mm: Union[float, List[float]] = 0.0,
        labels: List[int] = None
    ) -> "PointSet":
        """
        Return the coordinates as a siibra PointSet.

        Parameters
        ----------
        labels: List[int]
            Label coordinates with integers.
        sigma_mm: float or List[float], default: 0.0
            Optional standard deviation of point locations. By default, only the
            coordinates are passed on.
        """
        coordinates = self.data.iloc[:, :3].to_numpy()
        return PointSet(
            coordinates=coordinates,
            space=self.anchor.space,
            sigma_mm=sigma_mm,
            labels=labels
        )

    @property
    def boundingbox(self) -> "BoundingBox":
        return self.as_pointset().boundingbox

    @property
    def data(self) -> 'DataFrame':
        """
        Return a pandas DataFrame representing the coordinates and values
        associated with them. (x, y, z, value#0, value#1, ...)
        """
        if self._data_cached is None:
            self._data_cached = self._loader.get()
        return self._data_cached.copy()

    def plot(self, *args, backend='matplotlib', **kwargs):
        if self.data.shape[1] <= 3:
            raise NotImplementedError(
                "The point distribution does not contain any value data."
            )
        kind = kwargs.pop('kind', "hist")
        return self.data.iloc[:, 3:].plot(
            *args, backend=backend, kind=kind, **kwargs
        )

    def get_kde_volume(
        self,
        normalize: bool = True,
        **template_kwargs
    ) -> "Volume":
        """
        Get the kernel density estimate from the points using their average
        uncertainty on the reference space template the coordinates belongs to.

        Parameters
        ----------
        normalize: bool, optional
        template_kwargs:
            - variant
            - resolution_mm
            - voi

        Returns
        -------
        Volume
        """
        return from_pointset(
            points=self.as_pointset(),
            target=self.anchor.space.get_template(template_kwargs.pop("variant", None)),
            normalize=normalize,
            min_num_point=1,
            **template_kwargs
        )
