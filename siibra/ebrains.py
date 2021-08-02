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
from .retrieval import LazyHttpLoader,DECODERS
from os import environ
import re

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
        self._detail_loader = EbrainsLoader(
            query_id = 'interactiveViewerKgQuery-v1_0',
            instance_id = match.group(1),
            params = {'vocab': 'https://schema.hbp.eu/myQuery/'})

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

class KgToken:
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

KG_TOKEN = KgToken.instance()

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
        LazyHttpLoader.__init__(
            self, url, DECODERS['.json'], self.SC_MESSAGES, 
            headers=headers, params=params )