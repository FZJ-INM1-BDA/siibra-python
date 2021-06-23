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
from .. import retrieval

IGNORE_PROJECTS = [
  'Julich-Brain: cytoarchitectonic probabilistic maps of the human brain'
]

kg_feature_query_kwargs={
    'params': {
        'vocab': 'https://schema.hbp.eu/myQuery/'
    }
}

kg_feature_summary_kwargs={
    'org': 'minds',
    'domain': 'core',
    'schema': 'parcellationregion',
    'version': 'v1.0.0'
}

kg_feature_full_kwargs={
    'org': 'minds',
    'domain': 'core',
    'schema': 'dataset',
    'version': 'v1.0.0'
}

KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME = 'siibra-kg-feature-summary-0.0.1'
KG_REGIONAL_FEATURE_FULL_QUERY_NAME='interactiveViewerKgQuery-v1_0'
_SUMMARY_CACHE_FILENAME='ebrainsquery_summary_cache_name'

class EbrainsRegionalDataset(RegionalFeature):
    def __init__(self, region, id, name, embargo_status):
        self.region = region
        self.id = id
        self.name = name
        self.embargo_status = embargo_status
        self._detail = None

    @staticmethod
    def get_cache():
        json_resp=ebrains.execute_query_by_id(
            query_id=KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME,
            msg='Fetching summary data from ebrains. This will take some time.',
            **kg_feature_query_kwargs,
            **kg_feature_summary_kwargs)
        
        retrieval.save_cache(
            _SUMMARY_CACHE_FILENAME.encode('utf-8'),
            bytes(json.dumps(json_resp).encode()))

    @staticmethod
    def retrieve_cache():
        return retrieval.get_cache(_SUMMARY_CACHE_FILENAME.encode('utf-8'))

    @property
    def detail(self):
        if not self._detail:
            self._load()
        return self._detail

    def _load(self):
        if self.id is None:
            raise Exception('id is required')
        match=re.search(r"\/([a-f0-9-]+)$", self.id)
        if not match:
            raise Exception('id cannot be parsed properly')
        instance_id=match.group(1)
        result=ebrains.execute_query_by_id(
            query_id=KG_REGIONAL_FEATURE_FULL_QUERY_NAME, 
            instance_id=instance_id,
            msg=f"Retrieving details for '{self.name}' from EBRAINS...",
            **kg_feature_query_kwargs,**kg_feature_full_kwargs)
        self._detail = result

    def __str__(self):
        return self.name


class EbrainsRegionalFeatureExtractor(FeatureExtractor):
    _FEATURETYPE=EbrainsRegionalDataset
    def __init__(self, atlas):
        FeatureExtractor.__init__(self,atlas)

        try:
            cache=retrieval.get_cache(_SUMMARY_CACHE_FILENAME.encode('utf-8'))
            result=json.loads(cache)
            logger.debug(f"Retrieved cached ebrains results.")
        except FileNotFoundError:
        
            # potentially, using ebrains_id is a lot quicker
            # but if user selects region with higher level of hierarchy, this benefit may be offset by numerous http calls
            # even if they were cached...
            # ebrains_id=atlas.selected_region.attrs.get('fullId', {}).get('kg', {}).get('kgId', None)

            result=ebrains.execute_query_by_id(query_id=KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME,
                **kg_feature_query_kwargs,**kg_feature_summary_kwargs)
            
            retrieval.save_cache(
                _SUMMARY_CACHE_FILENAME.encode('utf-8'),
                bytes(json.dumps(result).encode()))
            logger.debug(f"Retrieved ebrain results via HTTP")

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
