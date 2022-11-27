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

from .feature import ParcellationFeature

from ..registry import REGISTRY, Preconfigure
from ..commons import logger, QUIET, create_key
from ..retrieval.repositories import RepositoryConnector

import pandas as pd
import hashlib
from io import BytesIO


# TODO: check if dataset_id is ebrains, then create the corresponding dataset
# in general, it should be cleaner to model for all features an attribute wether the come from an EBRAINS dataset,
# instead of implementing additional classes of objects which are ebrainsdatasets.
class ConnectivityMatrix(ParcellationFeature):

    """Connectivity matrix grouped by a parcellation."""

    def __init__(
        self,
        parcellation_id: str,
        cohort: str,
        connector: RepositoryConnector,
        datafile: str,
        headerfile: str,
        subject: str = None,
        dataset_id: str = None,
    ):
        """Construct a parcellation-averaged connectivty matrix."""
        ParcellationFeature.__init__(self, parcellation_id)
        self.cohort = cohort.upper()
        self.subject = subject
        self.dataset_id = dataset_id
        self._connector = connector
        self._datafile = datafile
        self._headerfile = headerfile
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
    def key(self):
        return create_key(f"{self.__str__()} {self.dataset_id}")

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
        parcellation = REGISTRY.Parcellation[self.spec]
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
                int(line[0]): parcellation.get_region(line[1])
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
            labels = {r.index.label for r in parcellation.regiontree} - {None}
            if max(labels) - min(labels) + 1 == nrows:
                indexmap = {
                    r.index.label - min(labels): r
                    for r in parcellation.regiontree
                    if r.index.label is not None
                }
                matrix = matrix.rename(index=indexmap).rename(columns=indexmap)
            else:
                logger.warn("Could not decode connectivity matrix regions.")

        return matrix


    @property
    def id(self):
        return f"siibra/features/connectivity/{hashlib.md5(str(self).encode('utf-8')).hexdigest()}"

    @id.setter
    def id(self, val):
        logger.warn(f"Connectivity matrix defines id as a property decorator")



@Preconfigure("features/connectivitymatrix/streamlinecounts")
class StreamlineCounts(ConnectivityMatrix):
    """Structural connectivity matrix of streamline counts grouped by a parcellation."""

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/connectivity/streamlineCounts"

    def __init__(
        self,
        parcellation_id: str,
        cohort: str,
        connector: RepositoryConnector,
        datafile: str,
        headerfile: str,
        subject: str = None,
        dataset_id: str = None,
    ):
        super().__init__(
            parcellation_id, cohort, connector, datafile, headerfile, subject, dataset_id
        )
        self.modality = "StreamlineCounts"

    @classmethod
    def _from_json(cls, spec):
        spectype = "siibra/resource/feature/connectivitymatrix/v1.0.0"
        assert spec.get("@type") == spectype
        conn = RepositoryConnector._from_json(spec['data']['repository'])
        return cls(
            parcellation_id=spec["parcellation_id"],
            cohort=spec["cohort"],
            connector=conn,
            datafile=spec['data']['datafile'],
            headerfile=spec['data']['headerfile'],
            subject=spec["subject"],
            dataset_id=spec["kgId"],
        )


@Preconfigure("features/connectivitymatrix/streamlinelengths")
class StreamlineLengths(ConnectivityMatrix):
    """Structural connectivity matrix of streamline lengths grouped by a parcellation."""

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/connectivity/streamlineLengths"

    def __init__(
        self,
        parcellation_id: str,
        cohort: str,
        connector: RepositoryConnector,
        datafile: str,
        headerfile: str,
        subject: str = None,
        dataset_id: str = None,
    ):
        super().__init__(
            parcellation_id, cohort, connector, datafile, headerfile, subject, dataset_id
        )
        self.modality = "StreamlineLengths"

    @classmethod
    def _from_json(cls, spec):
        spectype = "siibra/resource/feature/connectivitymatrix/v1.0.0"
        assert spec.get("@type") == spectype
        conn = RepositoryConnector._from_json(spec['data']['repository'])
        return cls(
            parcellation_id=spec["parcellation_id"],
            cohort=spec["cohort"],
            connector=conn,
            datafile=spec['data']['datafile'],
            headerfile=spec['data']['headerfile'],
            subject=spec["subject"],
            dataset_id=spec["kgId"],
        )


@Preconfigure("features/connectivitymatrix/functional")
class FunctionalConnectivity(ConnectivityMatrix):
    """Functional connectivity matrix, grouped by a parcellation."""

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/connectivity/functional"

    def __init__(
        self,
        paradigm: str,
        parcellation_id: str,
        cohort: str,
        connector: RepositoryConnector,
        datafile: str,
        headerfile: str,
        subject: str = None,
        dataset_id: str = None,
    ):
        super().__init__(
            parcellation_id, cohort, connector, datafile, headerfile, subject, dataset_id
        )
        self.modality = "FunctionalConnectivity"
        self.paradigm = paradigm

    @classmethod
    def _from_json(cls, spec):
        spectype = "siibra/resource/feature/connectivitymatrix/v1.0.0"
        assert spec.get("@type") == spectype
        conn = RepositoryConnector._from_json(spec['data']['repository'])
        return cls(
            parcellation_id=spec["parcellation_id"],
            cohort=spec["cohort"],
            connector=conn,
            datafile=spec['data']['datafile'],
            headerfile=spec['data']['headerfile'],
            subject=spec["subject"],
            dataset_id=spec["kgId"],
            paradigm=spec["paradigm"],
        )
