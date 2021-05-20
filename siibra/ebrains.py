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

from . import logger
from .retrieval import cached_get
import requests
import json
from os import environ

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

