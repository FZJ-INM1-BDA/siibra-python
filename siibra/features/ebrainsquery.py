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

import json
from .feature import RegionalFeature
from .extractor import FeatureExtractor
import re
import os
from .. import ebrains
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

KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME = 'siibra-kg-feature-summary-0.0.1'

class EbrainsRegionalDataset(RegionalFeature, ebrains.EbrainsDataset):
    def __init__(self, region, id, name, embargo_status):
        RegionalFeature.__init__(self, region)
        ebrains.EbrainsDataset.__init__(self, id, name, embargo_status)

    def __str__(self):
        super(ebrains.EbrainsDataset, self)

class EbrainsRegionalFeatureExtractor(FeatureExtractor):
    _FEATURETYPE=EbrainsRegionalDataset
    def __init__(self, atlas):
        FeatureExtractor.__init__(self,atlas)
        
        # potentially, using ebrains_id is a lot quicker
        # but if user selects region with higher level of hierarchy, this benefit may be offset by numerous http calls
        # even if they were cached...
        # ebrains_id=atlas.selected_region.attrs.get('fullId', {}).get('kg', {}).get('kgId', None)

        result=ebrains.execute_query_by_id(query_id=KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME,
            **ebrains.kg_feature_query_kwargs,**kg_feature_summary_kwargs)

        for r in result.get('results', []):
            for dataset in r.get('datasets', []):
                ds_id = dataset.get('@id')
                ds_name = dataset.get('name')
                if not "dataset" in ds_id:
                    logger.debug(f"'{ds_name}' is not an interpretable dataset and will be skipped.\n(id:{ds_id})")
                    continue
                regionname = r.get('name', None)
                try:
                    region = atlas.selected_parcellation.decode_region(regionname)
                except ValueError as e:
                    continue
                feature = EbrainsRegionalDataset(region=region, id=ds_id, name=ds_name,
                    embargo_status=dataset.get('embargo_status'))
                self.register(feature)


def set_specs():
    MODULE_DIR=os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(MODULE_DIR,'ebrainsquery_specs.json'),'r') as f:
        QUERYSPEC=json.load(f)
    req = ebrains.upload_schema(
        query_id=KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME,
        spec=QUERYSPEC['summary_spec'],
        **kg_feature_summary_kwargs)
    if not req.status_code < 400:
        raise RuntimeError("Could not upload query")

if __name__ == '__main__':
    pass
