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
from .query import FeatureQuery

from ..commons import logger, QUIET
from ..core.parcellation import Parcellation
from ..retrieval.repositories import GitlabConnector, EbrainsPublicDatasetConnector
from ..core.serializable_concept import NpArrayDataModel, JSONSerializable
from ..openminds.base import ConfigBaseModel

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import hashlib
from pydantic import Field


class ConnectivityMatrixDataModel(ConfigBaseModel):
    id: str = Field(..., alias="@id")
    type: str = Field(..., alias="@type")
    name: Optional[str]
    description: Optional[str]
    citation: Optional[str]
    authors: Optional[List[str]]
    cohort: Optional[str]
    subject: Optional[str]

    filename: Optional[str]
    dataset_id: Optional[str]

    parcellations: List[Dict[str, str]]
    matrix: Optional[NpArrayDataModel]
    columns: Optional[List[str]]


class ConnectivityMatrix(ParcellationFeature, JSONSerializable):

    """Structural connectivity matrix grouped by a parcellation."""

    def __init__(self, parcellation_id: str, matrixloader, srcinfo):
        """Construct a parcellation-averaged connectivty matrix.

        Arguments
        ---------
        parcellation_id : str
            Id of corresponding parcellation
        matrixloader : func
            Function which loads the matrix as a pandas dataframe
        """
        ParcellationFeature.__init__(self, parcellation_id)
        self._matrix_loader = matrixloader
        self.src_info = srcinfo

    def __getattr__(self, name):
        if name in self.src_info:
            return self.src_info[name]
        else:
            raise AttributeError(f"{self.__class__} has no attribute {name}")

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
        return self._matrix_loader()

    def get_profile(self, regionspec):
        for p in self.parcellations:
            region = p.decode_region(regionspec)
            return self.matrix[region]

    def __str__(self):
        return ParcellationFeature.__str__(self) + " " + str(self.src_info)

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/connectivity"

    @property
    def model_id(self):
        return f"{self.get_model_type()}/{hashlib.md5(str(self).encode('utf-8')).hexdigest()}"

    def to_model(self, detail=False, **kwargs) -> ConnectivityMatrixDataModel:

        # these properties do not exist on ConnectivityMatrix, but may be populated from ConnectivityMatrix.src_info via __getattr__
        model_dict_from_getattr = {
            key: getattr(self, key) if hasattr(self, key) else None
            for key in [
                "name",
                "description",
                "citation",
                "authors",
                "cohort",
                "subject",
                "filename",
                "dataset_id",
            ]
        }
        base_model = ConnectivityMatrixDataModel(
            id=self.model_id,
            type=self.get_model_type(),
            parcellations=[{
                "@id": parc.model_id,
            } for parc in self.parcellations],
            **model_dict_from_getattr,
        )
        if detail is False:
            return base_model

        from ..core import Region
        dtype_set = {dtype for dtype in self.matrix.dtypes}
        
        if len(dtype_set) == 0:
            raise TypeError(f"dtype is an empty set!")

        force_float = False
        if len(dtype_set) == 1:
            dtype, = list(dtype_set)
            is_int = np.issubdtype(dtype, int)
            is_float = np.issubdtype(dtype, float)
            assert is_int or is_float, f"expect datatype to be subdtype of either int or float, but is neither: {str(dtype)}"

        if len(dtype_set) > 1:
            logger.warning(f"expect only 1 type of data, but got {len(dtype_set)}, will cast everything to float")
            force_float = True
        
        def get_column_name(col: Union[str, Region]) -> str:
            if isinstance(col, str):
                return col
            if isinstance(col, Region):
                return col.name
            raise TypeError(f"matrix column value {col} of instance {col.__class__} can be be converted to str.")

        base_model.columns = [get_column_name(name) for name in self.matrix.columns.values]
        base_model.matrix=NpArrayDataModel(self.matrix.to_numpy(dtype="float32" if force_float or is_float else "int32"))
        return base_model


class StreamlineCounts(ConnectivityMatrix):
    """Structural connectivity matrix of streamline counts grouped by a parcellation."""

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/connectivity/streamlineCounts"

    def __init__(self, parcellation_id: str, matrixloader, srcinfo):
        super().__init__(parcellation_id, matrixloader, srcinfo)


class StreamlineLengths(ConnectivityMatrix):
    """Structural connectivity matrix of streamline lengths grouped by a parcellation."""
    
    @classmethod
    def get_model_type(Cls):
        return "siibra/features/connectivity/streamlineLengths"
    
    def __init__(self, parcellation_id: str, matrixloader, srcinfo):
        super().__init__(parcellation_id, matrixloader, srcinfo)


class FunctionalConnectivity(ConnectivityMatrix):
    """Functional connectivity matrix, grouped by a parcellation."""

    @classmethod
    def get_model_type(Cls):
        return "siibra/features/connectivity/functional"

    def __init__(self, parcellation_id: str, matrixloader, paradigm: str, srcinfo):
        super().__init__(parcellation_id, matrixloader, srcinfo)
        self.paradigm = paradigm


class HcpConnectivityFetcher:

    _DATASET_ID = "0f1ccc4a-9a11-4697-b43f-9c9c8ac543e6"
    # TODO add other parcellations
    _PARCELLATION_NAMES = {
        "294-Julich-Brain": Parcellation.REGISTRY.JULICH_BRAIN_CYTOARCHITECTONIC_MAPS_2_9,
        "096-HarvardOxfordMaxProbThr0": Parcellation.REGISTRY.HARVARDOXFORD_CORT_MAXPROB_THR0,
    }

    def __init__(self, filename_keyword):
        FeatureQuery.__init__(self)
        self._connector = EbrainsPublicDatasetConnector(
            self._DATASET_ID, in_progress=False
        )
        if len(self._connector._files) == 0:
            logger.error(
                f"EBRAINS knowledge graph query for dataset {self._DATASET_ID} "
                "did not return any files. It seems that your EBRAINS authentication "
                "does not provide sufficient privileges to access the connectivity data."
                )
        self._keyword = filename_keyword

    @property
    def doi(self):
        return self._connector.doi

    @property
    def srcinfo(self):
        return {
            "name": self._connector.name,
            "dataset_id": self._DATASET_ID, 
            "description": self._connector.description,
            "citation": self._connector.citation,
            "authors": self._connector.authors,
            "cohort": "HCP"}

    def get_matrixloaders(self, parcellation: Parcellation):
        """Return functions for loading the connectivity matrices
        matching the given parcellation as pandas DataFrames."""

        loaders = []
        for name, parc in self._PARCELLATION_NAMES.items():

            if parc != parcellation:
                continue

            try:
                zipfile = self._connector.get(f"{name}.zip")
            except RuntimeError as e:
                logger.error(str(e))
                continue

            # extract index - regionname mapping
            with zipfile.open(f"{name}/0ImageProcessing/Link.txt") as f:
                lines = [
                    line.decode().strip().split(" ", maxsplit=1)
                    for line in f.readlines()
                ]
                with QUIET:
                    indexmap = {
                        int(line[0]): parc.decode_region(line[1])
                        for line in lines
                        if len(line) == 2 and line[0].isnumeric()
                    }

            # create a dict of the csv files by subject id
            csvfiles = [
                f
                for f in zipfile.namelist()
                if f.endswith(".csv") and (self._keyword in f)
            ]

            for fn in csvfiles:
                subject_id = fn.split("/")[-2]
                # define the lazy loader function
                loaders.append(
                    (
                        fn,
                        subject_id,
                        lambda z=f"{name}.zip", c=fn, m=indexmap, p=parc: self._load_matrix(
                            z, c, m, p
                        ),
                    )
                )

        return loaders

    def _load_matrix(self, zip_filename, csv_filename, indexmap, parcellation):
        """Extract connectivity matrix from a csv file inside a zip file
        known to the dataset connector, and return it as a dataframe with
        region objects as row and column indices.
        """
        logger.debug(f"Loading {csv_filename} from {zip_filename}.")
        zipfile = self._connector.get(zip_filename)
        try:
            matrix = pd.read_csv(
                zipfile.open(csv_filename),
                delimiter=r"\s+|,|;", engine='python',
                header=None,
                index_col=False,
            )
            l = matrix.shape[0]
            if matrix.shape[1] != l:
                raise RuntimeError(
                    f"Non-quadratic connectivity matrix {l}x{matrix.shape[1]} "
                    f"from {csv_filename} in {zip_filename}"
                )
            if len(indexmap) == l:
                remapper = {label - min(indexmap.keys()): region for label, region in indexmap.items()}
                matrix = matrix.rename(index=remapper).rename(columns=remapper)
            else:
                labels = {r.index.label for r in parcellation.regiontree} - {None} 
                if max(labels) - min(labels) + 1 == l:
                    indexmap = {
                        r.index.label - min(labels): r for r in parcellation.regiontree
                        if r.index.label is not None
                    }
                    matrix = matrix.rename(index=indexmap).rename(columns=indexmap)
                else:
                    logger.warn("Could not decode connectivity matrix regions.")

        except pd.errors.ParserError:
            logger.error(
                f"Could not parse connectivity matrix from file {csv_filename} in {zip_filename}."
            )
        return matrix


    @property
    def parcellations(self):
        return list(self._PARCELLATION_NAMES.values())


class HcpStreamlineCountQuery(HcpConnectivityFetcher, FeatureQuery):

    _FEATURETYPE = StreamlineCounts

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        HcpConnectivityFetcher.__init__(self, filename_keyword="Counts")
        for parc in self.parcellations:
            for filename, subject_id, loader in self.get_matrixloaders(parc):
                srcinfo = self.srcinfo
                srcinfo["subject"] = subject_id
                srcinfo["filename"] = filename
                self.register(self._FEATURETYPE(parc, loader, srcinfo))


class HcpStreamlineLengthQuery(HcpConnectivityFetcher, FeatureQuery):

    _FEATURETYPE = StreamlineLengths

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        HcpConnectivityFetcher.__init__(self, filename_keyword="Lengths")
        for parc in self.parcellations:
            for filename, subject_id, loader in self.get_matrixloaders(parc):
                srcinfo = self.srcinfo
                srcinfo["subject"] = subject_id
                srcinfo["filename"] = filename
                self.register(self._FEATURETYPE(parc, loader, srcinfo))


class HcpRestingStateQuery(HcpConnectivityFetcher, FeatureQuery):

    _FEATURETYPE = FunctionalConnectivity

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        HcpConnectivityFetcher.__init__(self, filename_keyword="EmpCorrFC")
        for parc in self.parcellations:
            for filename, subject_id, loader in self.get_matrixloaders(parc):
                paradigm = "Resting state ({})".format(
                    filename.split("/")[-1].replace(".csv", "")
                )
                srcinfo = self.srcinfo
                srcinfo["subject"] = subject_id
                srcinfo["filename"] = filename
                self.register(self._FEATURETYPE(parc, loader, paradigm, srcinfo))


class PrereleasedConnectivityFetcher(GitlabConnector):

    _COHORTS = ["1000brains", "hcp", "enki"]

    def __init__(self, keywords: List[str]):
        """Gitlab connector for accessing some pre-released connectivity matrices."""
        GitlabConnector.__init__(self, "https://jugit.fz-juelich.de", 3009, "develop")
        self._keywords = keywords

    def get_data(self):
        results = []
        for filename, jsonloader in self.get_loaders("connectivity", ".json"):
            if not any(kw.lower() in filename.lower() for kw in self._keywords):
                continue
            parc_id = jsonloader.data["parcellation id"]
            matrixloader = lambda l=jsonloader: self._matrixloader(l)
            srcinfo = {k: v for k, v in jsonloader.data.items() if k != "data"}
            srcinfo["filename"] = filename
            matched_cohorts = [
                c for c in self._COHORTS if filename.lower().find(c.lower()) >= 0
            ]
            if len(matched_cohorts) > 0:
                srcinfo["cohort"] = matched_cohorts[0].upper()
                srcinfo["subject"] = "average"
            results.append((parc_id, matrixloader, srcinfo))
        return results

    def _matrixloader(self, jsonloader):
        """
        Load connectivity matrix from json object
        """
        data = jsonloader.data
        assert "data" in data
        col_names = data["data"]["field names"]
        row_names = list(data["data"]["profiles"].keys())
        assert col_names == row_names, f"{data['name']} assertion error: expected col_names == row_names"
        matrix = pd.DataFrame(
            data=[data["data"]["profiles"][r] for r in col_names],
            columns=col_names,
            index=col_names,
        )
        return matrix


class PrereleasedStreamlineCountQuery(FeatureQuery, PrereleasedConnectivityFetcher):

    _FEATURETYPE = StreamlineCounts

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        PrereleasedConnectivityFetcher.__init__(self, ["count", "1000brains"])
        for parc_id, matrixloader, srcinfo in self.get_data():
            self.register(self._FEATURETYPE(parc_id, matrixloader, srcinfo))


class PrereleasedStreamlineLengthQuery(FeatureQuery, PrereleasedConnectivityFetcher):

    _FEATURETYPE = StreamlineLengths

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        PrereleasedConnectivityFetcher.__init__(self, ["length"])
        for parc_id, matrixloader, srcinfo in self.get_data():
            self.register(self._FEATURETYPE(parc_id, matrixloader, srcinfo))


class PrereleasedRestingStateQuery(FeatureQuery, PrereleasedConnectivityFetcher):

    _FEATURETYPE = FunctionalConnectivity

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        keyword = "rest_fix"
        PrereleasedConnectivityFetcher.__init__(self, [keyword])
        for parc_id, matrixloader, srcinfo in self.get_data():
            filename = srcinfo["filename"]
            paradigm = filename[filename.lower().find(keyword.lower()):]
            self.register(self._FEATURETYPE(parc_id, matrixloader, paradigm, srcinfo))
