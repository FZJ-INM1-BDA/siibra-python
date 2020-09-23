import requests
import json

from brainscapes.authentication import Authentication
from brainscapes.retrieval import cached_get

authentication = Authentication.instance()


def upload_schema_from_file(file, org, domain, schema, version, query_id):
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}".format(
        org,
        domain,
        schema,
        version,
        query_id)
    r = requests.put(
        url,
        data=open(file, 'r'),
        headers={
            'Content-Type':'application/json',
            'Authorization': 'Bearer {}'.format(authentication.get_token())}
    )

    if r.status_code == 401:
        print('Not valid authentication')
    if r.status_code != 200:
        print('Some error while uploading the query appeared')


def execute_query_by_id(org, domain, schema, version, query_id, params=''):
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}/instances?databaseScope=RELEASED{}".format(
        org,
        domain,
        schema,
        version,
        query_id,
        params)
    r = cached_get(
        url,
        headers={
            'Content-Type':'application/json',
            'Authorization': 'Bearer {}'.format(authentication.get_token())})
    return json.loads(r)
        # results = json.loads(r)
        # for r in results['results']:
        #     receptors = ReceptorData(r)
        #     for name,dataframe in receptors.profiles.items():
        #         print(name,receptors.receptor_label[name])

