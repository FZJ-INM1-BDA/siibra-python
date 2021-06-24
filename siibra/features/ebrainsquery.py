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

from memoization import cached
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
    def preheat(parc_id: str = None):
        json_resp=get_dataset()
        from .. import parcellations

        results = json_resp.get('results', []) 
        if parc_id is not None:
            p=parcellations[parc_id]
            if p is not None:
                _ = [datasets_to_features(r, r.get('datasets', []), parcellation=p) for r in results]
        else:
            _ = [datasets_to_features(r, r.get('datasets', []), parcellation=p) for p in parcellations for r in results ]
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

@cached
def datasets_to_features(reg, datasets, parcellation):
    out = [dataset_to_feature(reg, ds, parcellation) for ds in datasets]
    return [f for f in out if f is not None ]

def dataset_to_feature(reg, dataset, parcellation):
    ds_id = dataset.get('@id')
    ds_name = dataset.get('name')
    if not "dataset" in ds_id:
        logger.debug(f"'{ds_name}' is not an interpretable dataset and will be skipped.\n(id:{ds_id})")
        return None
    regionname = reg.get('name', None)
    try:
        region = parcellation.decode_region(regionname)
    except ValueError:
        return None
    return EbrainsRegionalDataset(region=region, id=ds_id, name=ds_name,
        embargo_status=dataset.get('embargo_status'))

_cached_dataset = None
def get_dataset():
    global _cached_dataset
    if _cached_dataset is not None:
        return _cached_dataset
    try:
        from .. import retrieval
        cache=retrieval.get_cache(_SUMMARY_CACHE_FILENAME.encode('utf-8'))
        result=json.loads(cache)
        logger.debug(f"Retrieved cached ebrains results.")
        _cached_dataset=result
        return result
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
        _cached_dataset=result
        return result

class EbrainsRegionalFeatureExtractor(FeatureExtractor):
    _FEATURETYPE=EbrainsRegionalDataset
    def __init__(self, **kwargs):
        FeatureExtractor.__init__(self,**kwargs)

        if self.parcellation is None:
            raise ValueError('EbrainsRegionalFeatureExtractor requires parcellation as positional argument')

        result=get_dataset()

        results = result.get('results', []) 
        features=[]
        list_features=[datasets_to_features(r, r.get('datasets', []), parcellation=self.parcellation) for r in results]
        for f in list_features:
            features.extend(f)
        self.register_many(features)


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
