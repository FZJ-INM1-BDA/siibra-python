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

from .commons import OriginDataInfo
from . import logger
from .retrieval import cached_get,LazyLoader
import requests
import json
from os import environ
import re

kg_feature_full_kwargs={
    'org': 'minds',
    'domain': 'core',
    'schema': 'dataset',
    'version': 'v1.0.0'
}

kg_feature_query_kwargs={
    'params': {
        'vocab': 'https://schema.hbp.eu/myQuery/'
    }
}
KG_REGIONAL_FEATURE_FULL_QUERY_NAME='interactiveViewerKgQuery-v1_0'
KG_REGIONAL_FEATURE_SUMMARY_QUERY_NAME = 'siibra-kg-feature-summary-0.0.1'

class EbrainsDataset:
    def __init__(self, id, name, embargo_status):
        
        self.id = id
        self.name = name
        self.embargo_status = embargo_status
        self._detail = None
        if id is None:
            raise TypeError('Dataset id is required')
        match=re.search(r"([a-f0-9-]+)$",id)        
        if not match:
            raise ValueError('id cannot be parsed properly')
        self._detail_loader = LazyLoader(
            url=None,
            func=lambda:self._load_details(match.group(1)))

    @staticmethod
    def _load_details(instance_id):
        return execute_query_by_id(
            query_id=KG_REGIONAL_FEATURE_FULL_QUERY_NAME, 
            instance_id=instance_id,
            **kg_feature_query_kwargs,**kg_feature_full_kwargs)

    @property
    def detail(self):
        return self._detail_loader.data

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        # Check type
        if type(o) is not EbrainsDataset and not issubclass(type(o), EbrainsDataset):
            return False
        # Check id
        return self.id == o.id

class EbrainsOriginDataInfo(EbrainsDataset, OriginDataInfo):

    @property
    def urls(self):
        return [{
            'doi': f,
        }for f in self.detail.get('kgReference', [])]

    @urls.setter
    def urls(self, val):
        pass

    @property
    def description(self):
        return self.detail.get('description')
    
    @description.setter
    def description(self, val):
        pass

    @property
    def name(self):
        # to prevent inf loop, if _detail is undefined, return id
        if self._detail is None:
            return None
        return self.detail.get('name')

    @name.setter
    def name(self, value):
        pass

    def __init__(self, id):
        EbrainsDataset.__init__(self, id=id, name=None, embargo_status=None)

    def __hash__(self):
        return super(EbrainsDataset, self)

    def __str__(self):
        return super(EbrainsDataset, self)

class Authentication(object):
    """
    Implements the authentication to EBRAINS API with an authentication token. Uses a Singleton pattern.
    """
    _instance = None
    _authentication_token = ''

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def get_token(self):
        if self._authentication_token == '':
            try:
                self._authentication_token = environ['HBP_AUTH_TOKEN']
            except KeyError:
                logger.warning('An authentication token must be set as an environment variable: HBP_AUTH_TOKEN')
        return self._authentication_token

    def set_token(self, token):
        logger.info('Updating EBRAINS authentication token.')
        self._authentication_token = token


# convenience function that is importet at the package level
def set_token(token):
    auth = Authentication.instance()
    auth.set_token(token)

authentication = Authentication.instance()

def upload_schema(org, domain, schema, version, query_id, file=None, spec=None):
    """
    Upload a query schema to the EBRAINS knowledge graph.

    Parameters
    ----------
    org : str
        organisation
    domain : str
        domain
    schema : str
        schema
    version : str
        version
    query_id : str
        query_id - Used to execute query without specifiying the spec.
    file : str
        path to file of json to be uploaded (overrides spec, if both present)
    spec : dict
        spec to be uploaded (overriden by file, if both present)

    Yields
    ------
    requests.put object
    """
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}".format(
    org, domain, schema, version, query_id)
    headers={
        'Content-Type':'application/json',
        'Authorization': 'Bearer {}'.format(authentication.get_token())
    }
    if file is not None:
        return requests.put( url, data=open(file, 'r'),
                headers=headers )
    if spec is not None:
        return requests.put( url, json=spec,
                headers=headers )
    raise ValueError('Either file: str or spec: dict is needed for upload_schema method')

def upload_schema_from_file(file, org, domain, schema, version, query_id):
    """
    TODO needs documentation and cleanup
    """
    r = upload_schema(org, domain, schema, version, query_id, file=file)
    if r.status_code == 401:
        logger.error('Invalid authentication in EBRAINS')
    if r.status_code != 200:
        logger.error('Error while uploading EBRAINS Knowledge Graph query.')


def execute_query_by_id(org, domain, schema, version, query_id, instance_id=None, params={}, msg=None):
    """
    TODO needs documentation and cleanup
    """
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}/instances{}?databaseScope=RELEASED".format(
        org, domain, schema, version, query_id, '/' + instance_id if instance_id is not None else '' )
    if msg is None:
        msg = "No cached data - querying the EBRAINS Knowledge graph..."
    r = cached_get( url, headers={
        'Content-Type':'application/json',
        'Authorization': 'Bearer {}'.format(authentication.get_token())
        }, 
        msg_if_not_cached=msg,
        params=params)
    return json.loads(r)

