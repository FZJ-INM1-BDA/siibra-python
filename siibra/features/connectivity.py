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

from ..commons import logger, QUIET
from ..retrieval.repositories import RepositoryConnector

import pandas as pd
from io import BytesIO


class ConnectivityMatrix(Feature):
    """Connectivity matrix grouped by a parcellation."""

    def __init__(
        self,
        cohort: str,
        subject: str,
        measuretype: str,
        connector: RepositoryConnector,
        files: dict,
        anchor: "AnatomicalAnchor",
        description: str = "",
        datasets: list = [],
    ):
        """Construct a parcellation-averaged connectivty matrix."""
        Feature.__init__(
            self,
            measuretype=measuretype,
            description=description,
            anchor=anchor,
            datasets=datasets,
        )
        self.cohort = cohort.upper()
        self.subject = subject
        self._connector = connector
        self._datafile = files.get("data", "")
        self._headerfile = files.get("header", "")
        self._matrix_cached = None
        self.modality = None

    def __dir__(self):
        return sorted(
            set(
                dir(super(ConnectivityMatrix, self))
                + list(self.__dict__.keys())
                + list(self.src_info.keys())
            )
        )

    @property
    def matrix(self):
        # load and return the matrix
        if self._matrix_cached is None:
            self._matrix_cached = self._load_matrix()
        return self._matrix_cached

    def get_profile(self, regionspec):
        for p in self.parcellations:
            region = p.get_region(regionspec)
            return self.matrix[region]

    def __str__(self):
        return "{} connectivity matrix for {} from {} cohort {}".format(
            self.paradigm if hasattr(self, "paradigm") else self.modality,
            "_".join(p.name for p in self.parcellations),
            self.cohort,
            self.subject,
        )

    def _load_matrix(self):
        """
        Extract connectivity matrix.
        """
        parcellations = self.anchor.represented_parcellations()
        assert len(parcellations) == 1
        parc = next(iter(parcellations))
        loader = self._connector.get_loader(self._datafile, decode_func=lambda b: b)
        try:
            matrix = pd.read_csv(
                BytesIO(loader.data),
                delimiter=r"\s+|,|;",
                engine="python",
                header=None,
                index_col=False,
            )
        except pd.errors.ParserError:
            logger.error(
                f"Could not parse connectivity matrix from file {self._datafile} in {str(self._connector)}."
            )
        nrows = matrix.shape[0]
        if matrix.shape[1] != nrows:
            raise RuntimeError(
                f"Non-quadratic connectivity matrix {nrows}x{matrix.shape[1]} "
                f"from {self._datafile} in {str(self._connector)}"
            )

        loader = self._connector.get_loader(self._headerfile, decode_func=lambda b: b)
        lines = [
            line.decode().strip().split(" ", maxsplit=1)
            for line in BytesIO(loader.data).readlines()
        ]
        with QUIET:
            indexmap = {
                int(line[0]): parc.get_region(line[1], build_group=True)
                for line in lines
                if len(line) == 2 and line[0].isnumeric()
            }
        if len(indexmap) == nrows:
            remapper = {
                label - min(indexmap.keys()): region
                for label, region in indexmap.items()
            }
            matrix = matrix.rename(index=remapper).rename(columns=remapper)
        else:
            labels = {r.index.label for r in parc.regiontree} - {None}
            if max(labels) - min(labels) + 1 == nrows:
                indexmap = {
                    r.index.label - min(labels): r
                    for r in parc.regiontree
                    if r.index.label is not None
                }
                matrix = matrix.rename(index=indexmap).rename(columns=indexmap)
            else:
                logger.warn("Could not decode connectivity matrix regions.")

        return matrix


class StreamlineCounts(ConnectivityMatrix, configuration_folder="features/connectivitymatrix/streamlinecounts"):
    """Structural connectivity matrix of streamline counts grouped by a parcellation."""

    def __init__(
        self,
        cohort: str,
        subject: str,
        measuretype: str,
        connector: RepositoryConnector,
        files: dict,
        anchor: "AnatomicalAnchor",
        description: str = "",
        datasets: list = [],
    ):
        ConnectivityMatrix.__init__(
            self,
            cohort=cohort,
            subject=subject,
            measuretype=measuretype,
            connector=connector,
            files=files,
            anchor=anchor,
            description=description,
            datasets=datasets
        )


class FunctionalConnectivity(ConnectivityMatrix, configuration_folder="features/connectivitymatrix/functional"):
    """Functional connectivity matrix grouped by a parcellation."""

    def __init__(
        self,
        cohort: str,
        subject: str,
        measuretype: str,
        paradigm: str,
        connector: RepositoryConnector,
        files: dict,
        anchor: "AnatomicalAnchor",
        description: str = "",
        datasets: list = [],
    ):
        ConnectivityMatrix.__init__(
            self,
            cohort=cohort,
            subject=subject,
            measuretype=measuretype,
            connector=connector,
            files=files,
            anchor=anchor,
            description=description,
            datasets=datasets
        )
        self.paradigm = paradigm


class StreamlineLengths(ConnectivityMatrix, configuration_folder="features/connectivitymatrix/streamlinelengths"):
    """Structural connectivity matrix of streamline lengths grouped by a parcellation."""

    def __init__(
        self,
        cohort: str,
        subject: str,
        measuretype: str,
        connector: RepositoryConnector,
        files: dict,
        anchor: "AnatomicalAnchor",
        description: str = "",
        datasets: list = [],
    ):
        ConnectivityMatrix.__init__(
            self,
            cohort=cohort,
            subject=subject,
            measuretype=measuretype,
            connector=connector,
            files=files,
            anchor=anchor,
            description=description,
            datasets=datasets
        )
