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

from ...commons import logger, QUIET
from ...core import region as _region
from ...locations import pointset
from ...retrieval.repositories import RepositoryConnector

from typing import Callable, Dict, Union, List
import pandas as pd
import numpy as np
from tqdm import tqdm


class ParcellationBasedBOLD(
    tabular.SignalTable,
    configuration_folder="",
    category="signal"
):
    # TODO: write
    """
    BOLD
    """
    # TODO: write
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
        description: str = "",
        datasets: list = [],
        paradigm: str = "",
        timestep: str = "",
    ):
        """
        """
        tabular.SignalTable.__init__(
            self,
            modality=modality,
            description=description,
            anchor=anchor,
            datasets=datasets,
        )
        self.cohort = cohort.upper()
        self._connector = connector
        self._files = files
        self._decode_func = decode_func
        self.regions = regions
        self._matrices = {}
        self.paradigm = paradigm
        self.timestep = timestep

    @property
    def subjects(self):
        """
        Returns the subject identifiers for which matrices are available.
        """
        return list(self._files.keys())

    def get_matrix(self, subject: str = None):
        """
        Returns a matrix as a pandas dataframe.

        Parameters
        ----------
        subject: str, default: None
            Name of the subject (see ConnectivityMatrix.subjects for available names).
            If "mean" or None is given, the mean is taken in case of multiple
            available matrices.
        Returns
        -------
        pd.DataFrame
            A square matrix with region names as the column and row names.
        """
        assert len(self) > 0
        if (subject is None) and (len(self) > 1):
            # multiple matrices available, but no subject given - return mean matrix
            logger.info(
                f"No subject name supplied, returning mean connectivity across {len(self)} subjects. "
                "You might alternatively specify an individual subject."
            )
            if "mean" not in self._matrices:
                all_arrays = [
                    self._connector.get(fname, decode_func=self._decode_func)
                    for fname in tqdm(
                        self._files.values(),
                        total=len(self),
                        desc=f"Averaging {len(self)} connectivity matrices"
                    )
                ]
                self._matrices['mean'] = self._array_to_dataframe(np.stack(all_arrays).mean(0))
            return self._matrices['mean']
        if subject is None:
            subject = next(iter(self._files.keys()))
        if subject not in self._files:
            raise ValueError(f"Subject name '{subject}' not known, use one of: {', '.join(self._files)}")
        if subject not in self._matrices:
            self._matrices[subject] = self._load_matrix(subject)
        return self._matrices[subject]
    
    def _load_matrix(self, subject: str):
        """
        Extract timeseries.
        """
        assert subject in self.subjects
        array = self._connector.get(self._files[subject], decode_func=self._decode_func)
        return self._array_to_dataframe(array)
    
    def get_profile(
        self,
        region: Union[str, _region.Region],
        subject: str = None,
        min_connectivity: float = 0,
        max_rows: int = None
    ):
        """
        Extract a regional profile from the matrix, to obtain a tabular data feature
        with the connectivity as the single column.

        Rows will be sorted by descending connection strength.
        Regions with connectivity smaller than "min_connectivity" will be discarded.
        If max_rows is given, only the subset of regions with highest connectivity is returned.

        Parameters
        ----------
        region: str, Region
        subject: str, default: None
        min_connectivity: float, default: 0
        max_rows: int, default: None
        """
        matrix = self.get_matrix(subject)

        def matches(r1, r2):
            if isinstance(r1, tuple):
                return any(r.matches(r2) for r in r1)
            else:
                assert isinstance(r1, _region.Region)
                return r1.matches(r2)

        regions = [r for r in matrix.index if matches(r, region)]
        if len(regions) == 0:
            raise ValueError(f"Invalid region specificiation: {region}")
        elif len(regions) > 1:
            raise ValueError(f"Region specification {region} matched more than one profile: {regions}")
        else:
            name = \
                f"Averaged {self.modality}" if subject is None \
                else f"{self.modality}"
            series = matrix[regions[0]]
            last_index = len(series) - 1 if max_rows is None \
                else min(max_rows, len(series) - 1)
            return tabular.SignalTable(
                description=self.description,
                modality=f"{self.modality} {self.cohort}",
                anchor=_anchor.AnatomicalAnchor(
                    species=list(self.anchor.species)[0],
                    region=regions[0]
                ),
                data=(
                    series[:last_index]
                    .to_frame(name=name)
                    .query(f'`{name}` > {min_connectivity}')
                    .sort_values(by=name, ascending=False)
                    .rename_axis('Target regions')
                ),
                datasets=self.datasets
            )

    def __len__(self):
        return len(self._files)

    def __str__(self):
        return "{} connectivity for {} from {} cohort ({} matrices)".format(
            self.paradigm if hasattr(self, "paradigm") else self.modality,
            "_".join(p.name for p in self.anchor.parcellations),
            self.cohort,
            len(self._files),
        )

    def compute_centroids(self, space):
        """
        Computes the list of centroid coordinates corresponding to
        matrix rows, in the given reference space.

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
        Convert a numpy array with the connectivity matrix to
        a DataFrame with regions as column and row headers.
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