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

from .feature import RegionalFeature
from .query import FeatureQuery

from .. import logger
from ..core.datasets import EbrainsDataset
from ..retrieval.requests import EbrainsRequest


class EbrainsRegionalDataset(RegionalFeature, EbrainsDataset):
    def __init__(self, regionspec, kg_id, name, embargo_status):
        RegionalFeature.__init__(self, regionspec)
        EbrainsDataset.__init__(self, kg_id, name, embargo_status)

    @property
    def url(self):
        return (
            f"https://search.kg.ebrains.eu/instances/{self.id.split('/')[-1]}"
        )

    def __str__(self):
        return EbrainsDataset.__str__(self)

    def __hash__(self):
        return EbrainsDataset.__hash__(self)

    def __eq__(self, o: object) -> bool:
        return EbrainsDataset.__eq__(self, o)


class EbrainsRegionalFeatureQuery(FeatureQuery):
    _FEATURETYPE = EbrainsRegionalDataset

    def __init__(self, **kwargs):

        FeatureQuery.__init__(self)

        loader = EbrainsRequest(
            query_id="siibra-kg-feature-summary-0.0.1",
            schema="parcellationregion",
            params={"vocab": "https://schema.hbp.eu/myQuery/"},
        )

        for r in loader.data.get("results", []):
            for dataset in r.get("datasets", []):
                ds_id = dataset.get("@id")
                ds_name = dataset.get("name")
                ds_embargo_status = dataset.get("embargo_status")
                if "dataset" not in ds_id:
                    logger.debug(
                        f"'{ds_name}' is not an interpretable dataset and will be skipped.\n(id:{ds_id})"
                    )
                    continue
                regionname = r.get("name", None)
                self.register(
                    EbrainsRegionalDataset(
                        regionname, ds_id, ds_name, ds_embargo_status
                    )
                )

        # NOTE:
        # Potentially, using ebrains_id is a lot quicker, but if user selects region with higher level of hierarchy,
        # this benefit may be offset by numerous http calls even if they were cached.
        #       ebrains_id=atlas.selected_region.attrs.get('fullId', {}).get('kg', {}).get('kgId', None)
        #       for region, ds_id, ds_name, ds_embargo_status in get_dataset(atlas.selected_parcellation):
        #           feature = EbrainsRegionalDataset(region=region, id=ds_id, name=ds_name,
        #               embargo_status=ds_embargo_status)
        #       self.register(feature)
