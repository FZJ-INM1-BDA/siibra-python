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

from dataclasses import dataclass
from hashlib import md5
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import tabular
from ..feature import Compoundable
from ...commons import siibra_tqdm
from ...core.assignment import Qualification
from ...retrieval.requests import HttpRequest

if TYPE_CHECKING:
    from ..anchor import AnatomicalAnchor
    from ...core.region import Region
    from ...core.space import Space
    from ...locations.location import Location


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
        loader: HttpRequest = None,
        affine: Union[np.ndarray, List[List[float]]] = None,
        modality: str = None,
        description: str = "",
        datasets: list = None,
        id: str = None,
        prerelease: bool = False,
        data: pd.DataFrame = None,
    ):
        """
        Parameters
        ----------
        anchor : AnatomicalAnchor
            Its space is the reference space of the coordinates, after the
            affine is applied.
        loader : HttpRequest, optional
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
        data : DataFrame, optional
            The points in RAS+ millimetres in the reference space, instead of
            a loader. The affine is not applied to them.
        """
        if (loader is None) == (data is None):
            raise ValueError(f"A {self.__class__.__name__} needs either a loader or data.")
        tabular.Tabular.__init__(
            self,
            description=description,
            modality=modality or self._default_modality,
            anchor=anchor,
            data=data,
            datasets=[] if datasets is None else datasets,
            id=id,
            prerelease=prerelease,
        )
        if affine is not None:
            affine = np.asarray(affine, dtype=float)
            if affine.shape != (4, 4):
                raise ValueError(f"The affine must be a 4x4 matrix, not {affine.shape}.")
        self._loader = loader
        self._affine = None if data is not None else affine

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


class CellDistribution(
    PointDistribution,
    Compoundable,
    configuration_folder="features/tabular/point_distribution/cell_distribution",
    category="cellular",
):
    """
    Positions of cells of one subject, within one region or across a larger
    extent such as the whole brain.

    Like all point distributions, cell distributions are anchored to their
    space and, if they have one, their region: a region query finds the
    cells of matching regions, a space query all cells in that space.

    Cell distributions are compounded by subject: a query returns one
    compound per subject and dataset, with one element per region matching
    the query. The data of a compound has the cells of all its regions, with
    the region of each cell in a ``region`` column.
    """

    _default_modality = "cell distribution"
    _filter_attrs = ["modality", "subject", "region"]
    _compound_attrs = ["modality", "subject"]
    _compound_within_dataset = True

    def __init__(self, subject: str, **kwargs):
        """
        Parameters
        ----------
        subject : str
            Identifier of the subject the cells belong to.
        **kwargs
            See PointDistribution.
        """
        PointDistribution.__init__(self, **kwargs)
        self.subject = str(subject)

    @property
    def region(self) -> Union[str, None]:
        """
        The region the cells are in, as the anchor specifies it: a region or
        a parcellation, or None if the anchor has neither, e.g. for a
        whole-brain table anchored to its space only. Indexes the elements of
        a subject's compound.
        """
        return self.anchor._regionspec

    @property
    def name(self) -> str:
        return f"{super().name}, subject: {self.subject}"

    @classmethod
    def _merge_elements(cls, elements, description, modality, anchor):
        """The cells of all elements, with the region of each cell."""
        frames = [
            element.data.assign(region=element.region)
            for element in siibra_tqdm(
                elements,
                desc=f"Reading cell distributions of subject {elements[0].subject}",
                unit="region",
                disable=all(getattr(element._loader, "cached", True) for element in elements),
            )
        ]
        return cls(
            subject=elements[0].subject,
            anchor=anchor,
            modality=modality,
            description=description,
            datasets=list(dict.fromkeys(ds for element in elements for ds in element.datasets)),
            prerelease=all(element._prerelease for element in elements),
            data=pd.concat(frames, ignore_index=True),
        )


@dataclass(frozen=True)
class InjectionSite:
    """
    Where the tracer of a tracer distribution was injected. The regions, the
    location, both or neither may be known.
    """

    regions: Tuple["Region", ...] = ()
    """Regions the injection site is specified as; may be several."""

    location: Optional["Location"] = None
    """Location of the injection, e.g. a point, in its space."""


class TracerDistribution(
    PointDistribution,
    configuration_folder="features/tabular/point_distribution/tracer_distribution",
    category="connectivity",
):
    """
    Positions of tracer labels of one experiment.

    Besides its space, the anchor of a tracer distribution is its injection
    site, not where its labels are: a query for a region finds the
    experiments injected there, and their labels may lie anywhere in the
    brain. A space query finds all experiments in that space. Tracer
    distributions are not compounded; each experiment is a feature of its
    own.
    """

    _default_modality = "tracer distribution"

    @property
    def injection_site(self) -> InjectionSite:
        """Regions and location of the injection, as far as they are known."""
        return InjectionSite(
            regions=tuple(
                region for region, qualification in self.anchor.regions.items()
                if qualification == Qualification.EXACT
            ),
            location=self.anchor.location,
        )
