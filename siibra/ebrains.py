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

import requests
import json

from . import logger
from .authentication import Authentication
from .retrieval import cached_get

authentication = Authentication.instance()

def upload_schema(org, domain, schema, version, query_id, file=None, spec=None):
    """
    Upload a query schema to KG

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


def execute_query_by_id(org, domain, schema, version, query_id, instance_id=None, params={}):
    """
    TODO needs documentation and cleanup
    """
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}/instances{}?databaseScope=RELEASED".format(
        org, domain, schema, version, query_id, '/' + instance_id if instance_id is not None else '' )
    r = cached_get( url, headers={
        'Content-Type':'application/json',
        'Authorization': 'Bearer {}'.format(authentication.get_token())
        }, 
        msg_if_not_cached="No cached data. Will now run EBRAINS KG query. This may take a while...",
        params=params)
    return json.loads(r)

