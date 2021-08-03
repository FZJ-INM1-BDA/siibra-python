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

from .dataset import Dataset,OriginDataInfo
from . import logger
from .retrieval import LazyHttpLoader,DECODERS
from os import environ
import re

class EbrainsDataset(Dataset):

    def __init__(self, id, name, embargo_status=None):
        Dataset.__init__(self,id,name=name) # Name will be lazy-loaded on access
        self.embargo_status = embargo_status
        self._detail = None
        if id is None:
            raise TypeError('Dataset id is required')
        
        match=re.search(r"([a-f0-9-]+)$",id)        
        if not match:
            raise ValueError(f'{self.__class__.__name__} initialized with invalid id: {self.id}')
        self._detail_loader = EbrainsLoader(
            query_id = 'interactiveViewerKgQuery-v1_0',
            instance_id = match.group(1),
            params = {'vocab': 'https://schema.hbp.eu/myQuery/'})

    @property
    def detail(self):
        return self._detail_loader.data

    @property
    def urls(self):
        return [{'doi': f,} for f in self.detail.get('kgReference',[])]

    @property
    def description(self):
        return self.detail.get('description')

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, o: object) -> bool:
        if type(o) is not EbrainsDataset and not issubclass(type(o), EbrainsDataset):
            return False
        return self.id == o.id


class EbrainsOriginDataInfo(EbrainsDataset,OriginDataInfo):

    def __init__(self, id, name):
        EbrainsDataset.__init__(self, id=id, name=name )
        OriginDataInfo.__init__(self, name=name )


class EbrainsKgToken:
    """
    Simple handler for EBRAINS knowledge graph token.
    """
    _instance = None
    _value = ''

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def get(self):
        if self._value == '':
            try:
                self._value = environ['HBP_AUTH_TOKEN']
            except KeyError:
                logger.warning('No token defined for EBRAINS knowledge graph. Please set $HBP_AUTH_TOKEN')
        return self._value

    def set(self, token):
        logger.info('Updating EBRAINS authentication token.')
        self._value = token

KG_TOKEN = EbrainsKgToken.instance()


class EbrainsLoader(LazyHttpLoader):
    """
    Implements lazy loading of HTTP Knowledge graph queries.
    """

    SC_MESSAGES = {
        401 : 'The provided EBRAINS authentication token is not valid',
        403 : 'No permission to access the given query',
        404 : 'Query with this id not found'
    }

    server = "https://kg.humanbrainproject.eu"
    org = 'minds'
    domain = 'core'
    version = 'v1.0.0'

    def __init__(self,query_id,instance_id=None,schema='dataset',params={}):
        inst_tail = '/' + instance_id if instance_id is not None else ''
        self.schema = schema
        url = "{}/query/{}/{}/{}/{}/{}/instances{}?databaseScope=RELEASED".format(
            self.server, self.org, self.domain, self.schema, self.version, 
            query_id, inst_tail )
        headers = {
                'Content-Type':'application/json',
                'Authorization': f'Bearer {KG_TOKEN.get()}'
            }
        logger.debug(f"Initializing EBRAINS query '{query_id}'")
        LazyHttpLoader.__init__(
            self, url, DECODERS['.json'], self.SC_MESSAGES, 
            headers=headers, params=params )