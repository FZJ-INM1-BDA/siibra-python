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


from .feature import RegionalFeature, ParcellationFeature
from .query import FeatureQuery

from ..commons import logger
from ..core.parcellation import Parcellation
from ..core.datasets import EbrainsDataset, Dataset
from ..retrieval.repositories import GitlabConnector

from collections import defaultdict
import numpy as np


class ConnectivityMatrix(ParcellationFeature):
    def __init__(self, parcellation_id, matrix):
        ParcellationFeature.__init__(self, parcellation_id)
        self.matrix = matrix

    @property
    def array(self):
        """
        Returns the pure data array with connectivity information
        """
        return np.array([self.matrix[f] for f in self.matrix.dtype.names[1:]])

    @property
    def regionnames(self):
        return self.matrix.dtype.names[1:]

    @property
    def globalrange(self):
        return [
            fnc(
                fnc(self._matrix_loader.data[f])
                for f in self._matrix_loader.data.dtype.names[1:]
            )
            for fnc in [min, max]
        ]

    @classmethod
    def _from_json(cls, data):
        """
        Build a connectivity matrix from json object
        """
        parcellation = Parcellation.REGISTRY[data["parcellation id"]]

        # determine the valid brain regions defined in the file,
        # as well as their indices
        column_names = data["data"]["field names"]
        valid_regions = {}
        matchings = defaultdict(list)
        for i, name in enumerate(column_names):
            try:
                region = parcellation.decode_region(name, build_group=False)
                if region not in valid_regions.values():
                    valid_regions[i] = region
                    matchings[region].append(name)
                else:
                    logger.debug(
                        f"Region occured multiple times in connectivity dataset: {region}"
                    )
            except ValueError:
                continue

        profiles = []
        for i, region in valid_regions.items():
            regionname = column_names[i]
            profile = [region.name] + [
                data["data"]["profiles"][regionname][j] for j in valid_regions.keys()
            ]
            profiles.append(tuple(profile))
        fields = [("sourceregion", "U64")] + [
            (v.name, "f") for v in valid_regions.values()
        ]
        matrix = np.array(profiles, dtype=fields)
        assert all(N == len(valid_regions) for N in matrix.shape)

        if "kgId" in data:
            return EbrainsConnectivityMatrix(
                data["kgId"],
                data["parcellation id"],
                matrix,
                name=data["name"],
                description=data["description"],
            )
        else:
            return ExternalConnectivityMatrix(
                data["@id"],
                data["parcellation id"],
                matrix,
                name=data["name"],
                description=data["description"],
            )


class ExternalConnectivityMatrix(ConnectivityMatrix, Dataset):
    def __init__(self, id, parcellation_id, matrix, name, description):
        assert id is not None
        ConnectivityMatrix.__init__(self, parcellation_id, matrix)
        Dataset.__init__(self, id, description=description)
        self.name = name


class EbrainsConnectivityMatrix(ConnectivityMatrix, EbrainsDataset):
    def __init__(self, kg_id, parcellation_id, matrix, name, description):
        assert kg_id is not None
        ConnectivityMatrix.__init__(self, parcellation_id, matrix)
        self._description_cached = description
        EbrainsDataset.__init__(self, kg_id, name)


class ConnectivityProfile(RegionalFeature):

    show_as_log = True

    def __init__(self, regionspec: str, connectivitymatrix: ConnectivityMatrix, index, **kwargs):
        assert regionspec is not None
        RegionalFeature.__init__(self, regionspec, **kwargs)
        self._matrix_index = index
        self._matrix = connectivitymatrix

    @property
    def profile(self):
        return self._matrix.matrix[self._matrix_index]

    @property
    def description(self):
        return self._matrix.description

    @property
    def name(self):
        return self._matrix.name

    @property
    def regionnames(self):
        return self._matrix.regionnames

    @property
    def globalrange(self):
        return self._matrix.globalrange

    def __str__(self):
        return f"{self.__class__.__name__} from dataset '{self._matrix.name}' for {self.regionspec}"

    def decode(self, parcellation, minstrength=0, force=True):
        """
        Decode the profile into a list of connections strengths to real regions
        that match the given parcellation.
        If a column name for the profile cannot be decoded, a dummy region is
        returned if force==True, otherwise we fail.
        """
        decoded = (
            (strength, parcellation.decode_region(regionname, build_group=not force))
            for strength, regionname in zip(self.profile, self.regionnames)
            if strength > minstrength
        )
        return sorted(decoded, key=lambda q: q[0], reverse=True)


class ConnectivityProfileQuery(FeatureQuery):

    _FEATURETYPE = ConnectivityProfile
    _QUERY = GitlabConnector(
        "https://jugit.fz-juelich.de", 3009, "develop"
    )  # folder="connectivity"

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        for _, loader in self._QUERY.get_loaders("connectivity", ".json"):
            cm = ConnectivityMatrix._from_json(loader.data)
            for parcellation in cm.parcellations:
                species = [atlas.species for atlas in parcellation.atlases]
                for regionname in cm.regionnames:
                    region = parcellation.decode_region(regionname, build_group=False)
                    if region is None:
                        raise RuntimeError(
                            f"Could not decode region name {regionname} in {parcellation}"
                        )
                    self.register(ConnectivityProfile(region, cm, regionname, species=species))


class ConnectivityMatrixQuery(FeatureQuery):

    _FEATURETYPE = ConnectivityMatrix
    _CONNECTOR = GitlabConnector("https://jugit.fz-juelich.de", 3009, "develop")

    def __init__(self, **kwargs):
        FeatureQuery.__init__(self)
        for _, loader in self._CONNECTOR.get_loaders("connectivity", ".json"):
            matrix = ConnectivityMatrix._from_json(loader.data)
            self.register(matrix)
