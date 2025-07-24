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

from typing import Callable, TYPE_CHECKING, Union, List

import numpy as np

from . import tabular
from .. import anchor as _anchor
from ...retrieval.requests import HttpRequest
from ...locations import PointCloud
from ...volumes.volume import from_pointcloud

if TYPE_CHECKING:
    from ...volumes.volume import Volume
    from ...locations import BoundingBox
    from pandas import DataFrame


class PointDistribution(tabular.Tabular):
    """
    Represents a data frame with at least 3 columns (x, y, z, value#0, value#1,...)
    where the coordinates corresponds to a reference space,
    """

    def __init__(
        self,
        modality: str,
        subject: str,
        file_url: str,
        anchor: "_anchor.AnatomicalAnchor",
        transform: dict = None,
        description: str = "",
        decoder: Callable = None,
        datasets: list = [],
        id: str = None,
    ):
        tabular.Tabular.__init__(
            self,
            description=description,
            modality=modality,
            anchor=anchor,
            data=None,  # lazy loading below
            datasets=datasets,
            id=id,
        )
        self._transform = transform
        self._loader = HttpRequest(file_url, decoder)
        self._subject = subject

    @property
    def name(self):
        return self._subject + " - " + super().name

    def __len__(self):
        """Total number of coordinates."""
        return len(self.data)

    def as_pointcloud(
        self, sigma_mm: Union[float, List[float]] = 0.0, labels: List[int] = None
    ) -> "PointCloud":
        """
        Return the coordinates as a siibra PointCloud.

        Parameters
        ----------
        labels: List[int]
            Label coordinates with integers.
        sigma_mm: float or List[float], default: 0.0
            Optional standard deviation of point locations. By default, only the
            coordinates are passed on.
        """
        coordinates = self.data.loc[:, ["x", "y", "z"]].to_numpy()
        ptcld = PointCloud(
            coordinates=coordinates,
            space=self._transform["target_space_id"],
            sigma_mm=sigma_mm,
            labels=labels,
        )
        return ptcld

    @property
    def boundingbox(self) -> "BoundingBox":
        return self.as_pointcloud().boundingbox

    @property
    def data(self) -> "DataFrame":
        """
        Return a pandas DataFrame representing the coordinates and values
        associated with them. (x, y, z, value#0, value#1, ...)
        """
        if self._data_cached is None:
            self._data_cached = self._loader.get()
            if self._transform.get("affine") is not None:
                coords_homogeneous = np.c_[
                    self._data_cached.loc[:, ["x", "y", "z"]].to_numpy(),
                    np.ones(len(self._data_cached)),
                ]
                self._data_cached.loc[:, ["x", "y", "z"]] = np.dot(
                    self._transform["affine"], coords_homogeneous.T
                )[:3, :].T
        return self._data_cached.copy()

    def plot(self, *args, column_name: str, backend="matplotlib", **kwargs):
        kind = kwargs.pop("kind", "hist")
        return self.data.loc[:, column_name].plot(*args, backend=backend, kind=kind, **kwargs)

    def get_kde_volume(self, normalize: bool = True, **template_kwargs) -> "Volume":
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
        return from_pointcloud(
            points=self.as_pointcloud(),
            target=self.anchor.space.get_template(template_kwargs.pop("variant", None)),
            normalize=normalize,
            min_num_point=1,
            **template_kwargs,
        )


class CellDistribution(
    PointDistribution,
    configuration_folder="features/tabular/point_distribution/cell_distribution",
    category="cellular",
):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TracingConnectivityDistribution(
    PointDistribution,
    configuration_folder="features/tabular/point_distribution/tracing_connectivity_distribution",
    category="connectivity",
):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
