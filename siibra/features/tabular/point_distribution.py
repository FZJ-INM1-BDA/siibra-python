# Copyright 2018-2026
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
"""Coordinates of observations, such as cells, in a reference space."""

from hashlib import md5
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd

from . import tabular
from ...retrieval.requests import HttpRequest

if TYPE_CHECKING:
    from ..anchor import AnatomicalAnchor
    from ...core.space import Space


class PointDistribution(tabular.Tabular):
    """
    Coordinates of observations in a reference space.

    The anchor gives the reference space, and may add a region or location.
    Space queries find every point distribution in their space; region and
    location queries only those anchored to a matching region or location.

    The data has one row per point, with the coordinates in the columns
    ``x``, ``y`` and ``z``: millimetres in the physical coordinates of the
    reference space, oriented RAS+ (x towards the right, y anterior, z
    superior) like all coordinates in siibra. The configuration is trusted
    to provide these columns, e.g. through the ``names`` of its decoder.

    If an affine is given, it maps the coordinates as stored, e.g. voxel
    indices of an atlas, to these RAS+ millimetres. It is applied whenever
    the data is read; the file and its cached copy stay unchanged.
    """

    _default_modality = "point distribution"

    def __init__(
        self,
        anchor: "AnatomicalAnchor",
        loader: HttpRequest,
        affine: Union[np.ndarray, List[List[float]]] = None,
        modality: str = None,
        description: str = "",
        datasets: list = None,
        id: str = None,
        prerelease: bool = False,
    ):
        """
        Parameters
        ----------
        anchor : AnatomicalAnchor
            Its space is the reference space of the coordinates, after the
            affine is applied.
        loader : HttpRequest
            Loader of the table, e.g. from a repository connector. Nothing is
            read before the data is accessed.
        affine : array-like (4, 4), optional
            Transformation from the stored coordinates to RAS+ millimetres in
            the reference space. By default, the stored coordinates are
            taken to be RAS+ millimetres already.
        modality : str, optional
            What the points represent. Defaults to the modality of the
            feature type.
        description : str, default: ""
        datasets : list, optional
        id : str, optional
        prerelease : bool, default: False
        """
        tabular.Tabular.__init__(
            self,
            description=description,
            modality=modality or self._default_modality,
            anchor=anchor,
            data=None,  # read through the loader when accessed
            datasets=[] if datasets is None else datasets,
            id=id,
            prerelease=prerelease,
        )
        if affine is not None:
            affine = np.asarray(affine, dtype=float)
            if affine.shape != (4, 4):
                raise ValueError(f"The affine must be a 4x4 matrix, not {affine.shape}.")
        self._loader = loader
        self._affine = affine

    @property
    def space(self) -> "Space":
        """The reference space of the coordinates: the space of the anchor."""
        return self.anchor.space

    @property
    def affine(self) -> Union[np.ndarray, None]:
        """Transformation from the stored coordinates to RAS+ millimetres, if any."""
        return None if self._affine is None else self._affine.copy()

    @property
    def id(self) -> str:
        url = getattr(self._loader, "url", None)
        if self._id or url is None:
            return super().id
        prefix = next((ds.id + "--" for ds in self.datasets if hasattr(ds, "id")), "")
        # The file is linked exactly, so it identifies the feature.
        return prefix + md5(f"{self.__class__.__name__}\n{url}".encode("utf-8")).hexdigest()

    @property
    def data(self) -> pd.DataFrame:
        """
        The points, one row per point. The columns ``x``, ``y`` and ``z`` hold
        the coordinates in RAS+ millimetres in the reference space.
        """
        data = super().data
        if self._affine is not None:
            xyz = ["x", "y", "z"]
            coords = data[xyz].to_numpy(dtype=float)
            data[xyz] = coords @ self._affine[:3, :3].T + self._affine[:3, 3]
        return data

    def plot(self, *args, **kwargs):
        raise NotImplementedError(
            f"Plotting is not implemented for {self.__class__.__name__}. "
            "Its data holds the coordinates in the columns x, y and z."
        )
