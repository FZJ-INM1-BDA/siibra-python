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
"""One ReceptorDensityFingerprint per row of a master receptor table (rows = areas, columns = receptors)."""

import hashlib
from typing import List

import pandas as pd

from . import query as _query
from ..commons import Species
from ..core.parcellation import Parcellation
from ..features import anchor as _anchor
from ..features.tabular.receptor_density_fingerprint import ReceptorDensityFingerprint
from ..retrieval import requests, datasets

TABLE_URL = (
    "https://data-proxy.ebrains.eu/api/v1/public/buckets/d-de62abc1-7252-4774-9965-5040f5e8fb6b/"
    "receptor_density_per_neuron/receptors_per_neuron.csv"
)
PARCELLATION_ID = "minds/core/parcellationatlas/v1.0.0/e3235c039c6f54c3ba151568c829f117"
DATASET_VERSION_UUID = "de62abc1-7252-4774-9965-5040f5e8fb6b"
SPECIES = [Species.MACACA_MULATTA, Species.MACACA_FASCICULARIS]


class MacaqueReceptorFingerprintQuery(
    _query.LiveQuery, args=[], FeatureType=ReceptorDensityFingerprint
):

    _features: List[ReceptorDensityFingerprint] = None

    def __init__(self, **kwargs):
        _query.LiveQuery.__init__(self, **kwargs)

    @classmethod
    def _load(cls) -> List[ReceptorDensityFingerprint]:
        if cls._features is None:
            parcellation = Parcellation.registry().get(PARCELLATION_ID)
            table = requests.HttpRequest(TABLE_URL).get()
            assert isinstance(table, pd.DataFrame)
            cls._features = []
            for area, values in table.set_index(table.columns[0]).iterrows():
                try:
                    regionspec = area.replace("area_", "", 1).replace("_", " ")
                    _ = parcellation.get_region(regionspec)
                except ValueError:  # not (yet) published in the parcellation
                    continue
                data = pd.DataFrame({"mean": values})
                data.index.name = "receptor"
                digest = hashlib.md5(
                    (area + values.to_csv() + "ReceptorDensityFingerprint").encode()
                ).hexdigest()
                cls._features.append(
                    ReceptorDensityFingerprint(
                        tsvfile=None,
                        data=data,
                        anchor=_anchor.AnatomicalAnchor(species=SPECIES, region=regionspec),
                        datasets=[
                            datasets.EbrainsV3DatasetVersion(id=DATASET_VERSION_UUID)
                        ],
                        id=f"{DATASET_VERSION_UUID}--{digest}",
                        unit="fmol/neuron",
                    )
                )
        return cls._features

    def query(self, concept) -> List[ReceptorDensityFingerprint]:
        return [f for f in self._load() if f.matches(concept)]
