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

from ...core import region as _region
from .. import anchor as _anchor
from ...commons import QUIET, siibra_tqdm
from ...locations import pointset
from ...retrieval.repositories import RepositoryConnector
from ...retrieval.requests import HttpRequest

from typing import Callable, List, Union
import pandas as pd
import numpy as np


class RegionalTimeseriesActivity(tabular.Tabular, Compoundable):
    """
    Datasets that provide regional activity over time.
    """

    _filter_attrs = ["modality", "cohort", "subject"]
    _compound_attrs = ["modality", "cohort"]

    def __init__(
        self,
        cohort: str,
        modality: str,
        regions: list,
        connector: RepositoryConnector,
        decode_func: Callable,
        filename: str,
        anchor: _anchor.AnatomicalAnchor,
        timestep: str,
        description: str = "",
        datasets: list = [],
        subject: str = "average",
        id: str = None
    ):
        """
        """
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
            data=None,  # lazy loading below
            id=id
        )
        self.cohort = cohort.upper()
        if isinstance(connector, str) and connector:
            self._connector = HttpRequest(connector, decode_func)
        else:
            self._connector = connector
        self._filename = filename
        self._decode_func = decode_func
        self.regions = regions
        self._table = None
        self._subject = subject
        val, unit = timestep.split(" ")
        self.timestep = (float(val), unit)

    @property
    def subject(self):
        """Returns the subject identifiers for which the table represents."""
        return self._subject

    @property
    def name(self):
        return f"{self.subject} - " + super().name + f" cohort: {self.cohort}"

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns a table as a pandas dataframe where the index is a timeseries.
        """
        if self._table is None:
            self._load_table()
        return self._table.copy()

    @classmethod
    def _merge_elements(
        cls,
        elements: List["RegionalTimeseriesActivity"],
        description: str,
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
    ):
        assert len({f.cohort for f in elements}) == 1
        assert len({f.timestep for f in elements}) == 1
        merged = cls(
            cohort=elements[0].cohort,
            regions=elements[0].regions,
            connector=elements[0]._connector,
            decode_func=elements[0]._decode_func,
            filename="",
            timestep=" ".join(str(val) for val in elements[0].timestep),
            subject="average",
            description=description,
            modality=modality,
            anchor=anchor,
            **{"paradigm": "average"} if getattr(elements[0], "paradigm") else {}
        )
        if isinstance(elements[0]._connector, HttpRequest):
            getter = lambda elm: elm._connector.get()
        else:
            getter = lambda elm: elm._connector.get(elm._filename, decode_func=elm._decode_func)
        all_arrays = [
            getter(elm)
            for elm in siibra_tqdm(
                elements,
                total=len(elements),
                desc=f"Averaging {len(elements)} activity tables"
            )
        ]
        merged._table = elements[0]._arraylike_to_dataframe(
            np.stack(all_arrays).mean(0)
        )
        return merged

    def _load_table(self):
        """
        Extract the timeseries table.
        """
        if isinstance(self._connector, HttpRequest):
            array = self._connector.data
        else:
            array = self._connector.get(self._filename, decode_func=self._decode_func)
        self._table = self._arraylike_to_dataframe(array)

    def _arraylike_to_dataframe(self, array: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        if not isinstance(array, np.ndarray):
            array = array.to_numpy()
        ncols = array.shape[1]
        table = pd.DataFrame(
            array,
            index=pd.TimedeltaIndex(
                np.arange(0, array.shape[0]) * self.timestep[0],
                unit=self.timestep[1],
                name="time"
            )
        )
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parc = next(iter(parcellations))
        with QUIET:
            columnmap = {
                i: parc.get_region(regionname, allow_tuple=True)
                for i, regionname in enumerate(self.regions)
            }
        if len(columnmap) == ncols:
            remapper = {
                label - min(columnmap.keys()): region
                for label, region in columnmap.items()
            }
            table = table.rename(columns=remapper)

        return table

    def __str__(self):
        return self.name

    def compute_centroids(self, space):
        """
        Computes the list of centroid coordinates corresponding to
        dataframe columns, in the given reference space.

        Parameters
        ----------
        space: Space, str

        Returns
        -------
        list[tuple(float, float, float)]
        """
        result = []
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parcmap = next(iter(parcellations)).get_map(space)
        all_centroids = parcmap.compute_centroids()
        for regionname in self.regions:
            region = parcmap.parcellation.get_region(regionname, allow_tuple=True)
            if isinstance(region, tuple):  # deal with sets of matched regions
                found = [c for r in region for c in r if c.name in all_centroids]
            else:
                found = [r for r in region if r.name in all_centroids]
            assert len(found) > 0
            result.append(
                tuple(pointset.PointSet(
                    [all_centroids[r.name] for r in found], space=space
                ).centroid)
            )
        return result

    def plot(
        self, regions: List[Union[str, "_region.Region"]] = None, *args,
        backend="matplotlib", **kwargs
    ):
        """
        Create a bar plot of averaged timeseries data per region.

        Parameters
        ----------
        regions: List[str or Region]
        subject: str, default: None
            If None, returns the subject averaged table.
        args and kwargs:
            takes arguments and keyword arguments for the desired plotting
            backend.
        """
        if isinstance(regions, (str, _region.Region)):
            regions = [regions]
        if regions is None:
            regions = self.regions
        indices = [self.regions.index(r) for r in regions]
        table = self.data.iloc[:, indices]
        table.columns = [str(r) for r in table.columns]
        return table.mean().plot(kind="bar", *args, backend=backend, **kwargs)

    def plot_carpet(
        self, regions: List[Union[str, "_region.Region"]] = None, *args,
        backend="plotly", **kwargs
    ):
        """
        Create a carpet plot ofthe timeseries data per region.

        Parameters
        ----------
        regions: List[str or Region]
        subject: str, default: None
            If None, returns the subject averaged table.
        args and kwargs:
            takes arguments and keyword arguments for `plotly.express.imshow`
        """
        if backend != "plotly":
            raise NotImplementedError("Currently, carpet plot is only implemented with `plotly`.")
        if isinstance(regions, (str, _region.Region)):
            regions = [regions]
        if regions is None:
            regions = self.regions
        indices = [self.regions.index(r) for r in regions]
        table = self.data.iloc[:, indices].reset_index(drop=True)
        table.columns = [str(r) for r in table.columns]
        kwargs["title"] = kwargs.get("title", f"{self.modality}" + f" for subject={self.subject}")
        kwargs["labels"] = kwargs.get("labels", {
            "xlabel": self.data.index.to_numpy(dtype='timedelta64[ms]')}
        )
        from plotly.express import imshow
        return imshow(
            *args,
            table.T,
            **kwargs
        )


class RegionalBOLD(
    RegionalTimeseriesActivity,
    configuration_folder="features/tabular/activity_timeseries/bold",
    category="functional"
):
    """
    Blood-oxygen-level-dependent (BOLD) signals per region.
    """

    _filter_attrs = RegionalTimeseriesActivity._filter_attrs + ["paradigm"]
    _compound_attrs = RegionalTimeseriesActivity._compound_attrs + ["paradigm"]

    def __init__(self, paradigm: str, **kwargs):
        RegionalTimeseriesActivity.__init__(self, **kwargs)
        self.paradigm = paradigm

        # paradign is used to distinguish functional connectivity features from each other.
        assert self.paradigm, "RegionalBOLD must have paradigm defined!"

    @property
    def name(self):
        return super().name + f", paradigm: {self.paradigm}"
