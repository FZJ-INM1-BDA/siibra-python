# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO this file lacks documentation

from .feature import RegionalFeature
from .query import FeatureQuery
import json
import os
from .. import ebrains
from ..ebrains import EbrainsDataset
from .. import logger


IGNORE_PROJECTS = [
  'Julich-Brain: cytoarchitectonic probabilistic maps of the human brain'
]

kg_feature_summary_kwargs={
    'org': 'minds',
    'domain': 'core',
    'schema': 'parcellationregion',
    'version': 'v1.0.0'
}

class EbrainsRegionalDataset(RegionalFeature,EbrainsDataset):

    def __init__(self, regionspec, id, name, embargo_status):
        RegionalFeature.__init__(self, regionspec,dataset_id=id)
        EbrainsDataset.__init__(self, id, name, embargo_status)

    def __str__(self):
        return EbrainsDataset.__str__(self)

    def __hash__(self):
        return EbrainsDataset.__hash__(self)

    def __eq__(self, o: object) -> bool:
        return EbrainsDataset.__eq__(self,o)

    @property
    def url(self):
        return f"https://search.kg.ebrains.eu/instances/Dataset/{self.dataset_id.split('/')[-1]}"


class EbrainsRegionalFeatureQuery(FeatureQuery):
    _FEATURETYPE=EbrainsRegionalDataset

    def __init__(self):
        FeatureQuery.__init__(self)

        for r in ebrains.execute_query_by_id(
            query_id=ebrains.KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME,
            **ebrains.kg_feature_query_kwargs, **kg_feature_summary_kwargs).get('results', []):
            for dataset in r.get('datasets', []):
                ds_id = dataset.get('@id')
                ds_name = dataset.get('name')
                ds_embargo_status=dataset.get('embargo_status')
                if not "dataset" in ds_id:
                    logger.debug(f"'{ds_name}' is not an interpretable dataset and will be skipped.\n(id:{ds_id})")
                    continue
                regionname = r.get('name', None)
                self.register(EbrainsRegionalDataset(regionname,ds_id,ds_name,ds_embargo_status))
        
        ## potentially, using ebrains_id is a lot quicker
        ## but if user selects region with higher level of hierarchy, this benefit may be offset by numerous http calls
        ## even if they were cached...
        ## ebrains_id=atlas.selected_region.attrs.get('fullId', {}).get('kg', {}).get('kgId', None)
        #for region, ds_id, ds_name, ds_embargo_status in get_dataset(atlas.selected_parcellation):
        #    feature = EbrainsRegionalDataset(region=region, id=ds_id, name=ds_name,
        #        embargo_status=ds_embargo_status)
        #    self.register(feature)


def set_specs():
    MODULE_DIR=os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(MODULE_DIR,'ebrainsquery_specs.json'),'r') as f:
        QUERYSPEC=json.load(f)
    req = ebrains.upload_schema(
        query_id=ebrains.KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME,
        spec=QUERYSPEC['summary_spec'],
        **kg_feature_summary_kwargs)
    if not req.status_code < 400:
        raise RuntimeError("Could not upload query")

if __name__ == '__main__':
    pass
