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

from .feature import Feature
from .tabular import Tabular

from .. import anchor as _anchor

from ...commons import logger, QUIET
from ...core import region as _region
from ...retrieval.repositories import RepositoryConnector

from typing import Callable, Dict, Union
import pandas as pd
import numpy as np
from tqdm import tqdm


class RegionalConnectivity(Feature):
    """
    Parcellation-averaged connectivity, providing one or more
    matrices of a given modality for a given parcellation.

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
        description: str = "",
        datasets: list = [],
    ):
        """j
        Construct a parcellation-averaged connectivty matrix.

        Parameters
        ----------
        cohort: str
            Name of the cohort used for computing the connectivity.
        modality: str
            Connectivity modality, typically set by derived classes.
        regions: list of str
            Names of the regions from the parcellation
        connector: Connector
            Repository connector for loading the actual data array(s).
        decode_func: function
            Function to convert the bytestream of a loaded file into an array
        files: dict
            A dictionary linking names of matrices (typically subject ids)
            to the relative filenames of the data array(s) in the repository connector.
        anchor: AnatomicalAnchor
            anatomical localization of the matrix, expected to encode the parcellation
            in the region attribute.
        description: str (optional)
            textual description of this connectivity matrix.
        datasets : list
            list of datasets corresponding to this feature.
        """
        Feature.__init__(
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
        subject: str
            Name of the subject (see ConnectivityMatrix.subjects for available names).
            If "mean" or None is given, the mean is taken in case of multiple
            available matrices.
        """
        assert len(self) > 0
        if (subject is None) and (len(self) > 1):
            # multiple matrices available, but no subject given - return mean matrix
            logger.info(
                f"No subject name supplied, returning mean connectivity across {len(self)} subjects. "
                "You might alternatively specifiy an individual subject."
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

    def __iter__(self):
        return ((sid, self.get_matrix(sid)) for sid in self._files)

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
        """
        matrix = self.get_matrix(subject)
        regions = [r for r in matrix.index if r.matches(region)]
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
            return Tabular(
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

    def _array_to_dataframe(self, array: np.ndarray) -> pd.DataFrame:
        """
        Convert a numpy array with the connectivity matrix to
        a dataframe with regions as column and row headers.
        """
        df = pd.DataFrame(array)
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parc = next(iter(parcellations))
        with QUIET:
            indexmap = {
                i: parc.get_region(regionname, build_group=True)
                for i, regionname in enumerate(self.regions)
            }
        nrows = array.shape[0]
        if len(indexmap) == nrows:
            remapper = {
                label - min(indexmap.keys()): region
                for label, region in indexmap.items()
            }
            df = df.rename(index=remapper).rename(columns=remapper)
        else:
            labels = {r.index.label for r in parc.regiontree} - {None}
            if max(labels) - min(labels) + 1 == nrows:
                indexmap = {
                    r.index.label - min(labels): r
                    for r in parc.regiontree
                    if r.index.label is not None
                }
                df = df.rename(index=indexmap).rename(columns=indexmap)
            else:
                logger.warn("Could not decode connectivity matrix regions.")
        return df

    def _load_matrix(self, subject: str):
        """
        Extract connectivity matrix.
        """
        assert subject in self.subjects
        array = self._connector.get(self._files[subject], decode_func=self._decode_func)
        nrows = array.shape[0]
        if array.shape[1] != nrows:
            raise RuntimeError(
                f"Non-quadratic connectivity matrix {nrows}x{array.shape[1]} "
                f"from {self._files[subject]} in {str(self._connector)}"
            )
        return self._array_to_dataframe(array)
