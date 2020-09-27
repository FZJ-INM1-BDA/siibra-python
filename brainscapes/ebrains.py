import requests
import json

from . import logger
from .authentication import Authentication
from .retrieval import cached_get

authentication = Authentication.instance()


def upload_schema_from_file(file, org, domain, schema, version, query_id):
    """
    TODO needs documentation and cleanup
    """
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}".format(
        org, domain, schema, version, query_id)
    r = requests.put( url, data=open(file, 'r'),
            headers={
                'Content-Type':'application/json',
                'Authorization': 'Bearer {}'.format(authentication.get_token())
                } )
    if r.status_code == 401:
        logger.error('Invalid authentication in EBRAINS')
    if r.status_code != 200:
        logger.error('Error while uploading EBRAINS Knowledge Graph query.')


def execute_query_by_id(org, domain, schema, version, query_id, params=''):
    """
    TODO needs documentation and cleanup
    """
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}/instances?databaseScope=RELEASED{}".format(
        org, domain, schema, version, query_id, params)
    r = cached_get( url, headers={
            'Content-Type':'application/json',
            'Authorization': 'Bearer {}'.format(authentication.get_token())
            } )
    return json.loads(r)

