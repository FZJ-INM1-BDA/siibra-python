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

from ...commons import logger, QUIET, siibra_tqdm
from ...locations import pointset
from ...retrieval.repositories import RepositoryConnector

from typing import Callable, Dict, List
import pandas as pd
import numpy as np


class RegionalTimeseriesActivity(tabular.Tabular):
    """
    Datasets that provide regional activity over time.
    """

    DESCRIPTION = (
        ""
    )

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
        prerelease: bool = False,
        id: str = None,
    ):
        """
        """
        tabular.Tabular.__init__(
            self,
            modality=modality,
            description=description or '\n'.join({ds.description for ds in datasets}),
            anchor=anchor,
            datasets=datasets,
            data=None,  # lazy loading below
            prerelease=prerelease,
            id=id,
        )
        self.cohort = cohort.upper()
        self._connector = connector
        self._files = files
        self._decode_func = decode_func
        self.regions = regions
        self._tables = {}
        self.paradigm = paradigm
        self.timestep = timestep

    @property
    def subjects(self):
        """
        Returns the subject identifiers for which signal tables are available.
        """
        return list(self._files.keys())

    @property
    def name(self):
        supername = super().name
        return f"{supername} with paradigm {self.paradigm}"

    def get_table(self, subject: str = None):
        """
        Returns a pandas dataframe where the column headers are regions and the
        indcies indicate disctrete timesteps.

        Parameters
        ----------
        subject: str, default: None
            Name of the subject (see RegionalTimeseriesActivity.subjects for available names).
            If None, the mean is taken in case of multiple available data tables.
        Returns
        -------
        pd.DataFrame
            A table with region names as the column and timesteps as indices.
        """
        assert len(self) > 0
        if (subject is None) and (len(self) > 1):
            # multiple signal tables available, but no subject given - return mean table
            logger.info(
                f"No subject name supplied, returning mean signal table across {len(self)} subjects. "
                "You might alternatively specify an individual subject."
            )
            if "mean" not in self._tables:
                all_arrays = [
                    self._connector.get(fname, decode_func=self._decode_func)
                    for fname in siibra_tqdm(
                        self._files.values(),
                        total=len(self),
                        desc=f"Averaging {len(self)} signal tables"
                    )
                ]
                self._tables['mean'] = self._array_to_dataframe(np.stack(all_arrays).mean(0))
            return self._tables['mean'].copy()
        if subject is None:
            subject = next(iter(self._files.keys()))
        if subject not in self._files:
            raise ValueError(f"Subject name '{subject}' not known, use one of: {', '.join(self._files)}")
        if subject not in self._tables:
            self._tables[subject] = self._load_table(subject)
        return self._tables[subject].copy()

    def _load_table(self, subject: str):
        """
        Extract the timeseries table.
        """
        assert subject in self.subjects
        array = self._connector.get(self._files[subject], decode_func=self._decode_func)
        return self._array_to_dataframe(array)

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
        df = pd.DataFrame(array)
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parc = next(iter(parcellations))
        with QUIET:
            indexmap = {
                i: parc.get_region(regionname, allow_tuple=True)
                for i, regionname in enumerate(self.regions)
            }
        ncols = array.shape[1]
        if len(indexmap) == ncols:
            remapper = {
                label - min(indexmap.keys()): region
                for label, region in indexmap.items()
            }
            df = df.rename(columns=remapper)
        return df

    def plot(
        self, subject: str = None, regions: List[str] = None, *args,
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
        table = self.get_table(subject).iloc[:, indices]
        table.columns = [str(r) for r in table.columns]
        return table.mean().plot(kind="bar", *args, backend=backend, **kwargs)

    def plot_carpet(
        self, subject: str = None, regions: List[str] = None, *args,
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
        table = self.get_table(subject).iloc[:, indices]
        table.columns = [str(r) for r in table.columns]
        from plotly.express import imshow
        return imshow(
            table.T,
            title=f"{self.modality}" + f" for subject={subject}" if subject else ""
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
