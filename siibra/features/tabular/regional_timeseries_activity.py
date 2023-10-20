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
from ...commons import QUIET, siibra_tqdm
from ...locations import pointset
from ...retrieval.repositories import RepositoryConnector

from typing import Callable, Dict, List
import pandas as pd
import numpy as np


class RegionalTimeseriesActivity(tabular.Tabular, Compoundable):
    """
    Datasets that provide regional activity over time.
    """

    def __init__(
        self,
        cohort: str,
        modality: str,
        regions: list,
        connector: RepositoryConnector,
        decode_func: Callable,
        files: Dict[str, str],
        anchor: _anchor.AnatomicalAnchor,
        timestep: str,
        description: str = "",
        datasets: list = [],
        paradigm: str = "",
        subject: str = "average"
    ):
        """
        """
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description or '\n'.join({ds.description for ds in datasets}),
            anchor=anchor,
            datasets=datasets,
            data=None  # lazy loading below
        )
        self.cohort = cohort.upper()
        self._connector = connector
        self._files = files
        self._decode_func = decode_func
        self.regions = regions
        self._table = None
        self.paradigm = paradigm
        self._subject = subject
        val, unit = timestep.split(" ")
        self.timestep = (float(val), unit)

    @property
    def filter_attributes(self) -> Dict[str, str]:
        return {
            attr: getattr(self, attr)
            for attr in ["modality", "cohort", "index", "paradigm"]
        }

    @property
    def _compound_key(self):
        return tuple((
            self.filter_attributes[attr]
            for attr in ["modality", "cohort"]
        ),)

    @property
    def subfeature_index(self) -> str:
        return (self.index, self.paradigm)

    @property
    def index(self):
        return list(self._files.keys())[0]

    @property
    def subject(self):
        """
        Returns the subject identifiers for which matrices are available.
        """
        return self._subject

    @property
    def name(self):
        supername = super().name
        postfix = f" and paradigm {self.paradigm}" if hasattr(self, 'paradigm') else ""
        return f"{supername} with cohort {self.cohort}" + postfix + f" - {self.index}"

    @property
    def data(self):
        """
        Returns a table as a pandas dataframe.

        Returns
        -------
        pd.DataFrame
            A square table with region names as the column and row names.
        """
        if self._table is None:
            self._table = self._load_table()
        return self._table.copy()

    @classmethod
    def _merge_instances(
        cls,
        instances: List["RegionalTimeseriesActivity"],
        description: str,
        modality: str,
        anchor: _anchor.AnatomicalAnchor,
    ):
        assert len({f.cohort for f in instances}) == 1
        assert len({f.timestep for f in instances}) == 1
        compounded = cls(
            timestep=f"{instances[0].timestep[0]} {instances[0].timestep[1]}",
            cohort=instances[0].cohort,
            regions=instances[0].regions,
            connector=instances[0]._connector,
            decode_func=instances[0]._decode_func,
            files=[],
            subject="average",
            description=description,
            modality=modality,
            anchor=anchor,
            paradigm="average"
        )
        all_arrays = [
            instance._connector.get(fname, decode_func=instance._decode_func)
            for instance in siibra_tqdm(
                instances,
                total=len(instances),
                desc=f"Averaging {len(instances)} connectivity matrices"
            )
            for fname in instance._files.values()
        ]
        compounded._table = compounded._array_to_dataframe(
            np.stack(all_arrays).mean(0)
        )
        return compounded

    def _load_table(self):
        """
        Extract the timeseries table.
        """
        array = self._connector.get(self._files[self.index], decode_func=self._decode_func)
        return self._array_to_dataframe(array.to_numpy())

    def __len__(self):
        return len(self._files)

    def __str__(self):
        return "{} with paradigm {} for {} from {} cohort ({} signal tables)".format(
            self.modality, self.paradigm,
            "_".join(p.name for p in self.anchor.parcellations),
            self.cohort,
            len(self._files),
        )

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

    def _array_to_dataframe(self, array: np.ndarray) -> pd.DataFrame:
        """
        Convert a numpy array with the regional activity data to
        a DataFrame with regions as column headers and timesteps as indices.
        """
        ncols = array.shape[1]
        df = pd.DataFrame(
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
            df = df.rename(columns=remapper)
        return df

    def plot(
        self, regions: List[str] = None, *args,
        backend="matplotlib", **kwargs
    ):
        """
        Create a bar plot of averaged timeseries data per region.

        Parameters
        ----------
        regions: str, Region
        subject: str, default: None
            If None, returns the subject averaged table.
        args and kwargs:
            takes arguments and keyword arguments for the desired plotting
            backend.
        """
        if regions is None:
            regions = self.regions
        indices = [self.regions.index(r) for r in regions]
        table = self.data.iloc[:, indices]
        table.columns = [str(r) for r in table.columns]
        return table.mean().plot(kind="bar", *args, backend=backend, **kwargs)

    def plot_carpet(
        self, regions: List[str] = None, *args,
        backend="plotly", **kwargs
    ):
        """
        Create a carpet plot ofthe timeseries data per region.

        Parameters
        ----------
        regions: str, Region
        subject: str, default: None
            If None, returns the subject averaged table.
        args and kwargs:
            takes arguments and keyword arguments for `plotly.express.imshow`
        """
        if backend != "plotly":
            raise NotImplementedError("Currently, carpet plot is only implemented with `plotly`.")
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

    pass
