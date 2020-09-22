import requests
import json

from brainscapes.authentication import Authentication
from brainscapes.retrieval import cached_get

#access_token = 'eyJhbGciOiJSUzI1NiIsImtpZCI6ImJicC1vaWRjIn0.eyJleHAiOjE2MDA0NDAyNTEsInN1YiI6IjMwODExMCIsImF1ZCI6WyJuZXh1cy1rZy1zZWFyY2giXSwiaXNzIjoiaHR0cHM6XC9cL3NlcnZpY2VzLmh1bWFuYnJhaW5wcm9qZWN0LmV1XC9vaWRjXC8iLCJqdGkiOiI0NmI3M2RlMi0yYzBhLTQxMWYtOWY3MC1iYWNmOTM3N2ZiMGEiLCJpYXQiOjE2MDA0MjU4NTIsImhicF9rZXkiOiI3M2VjYjRhNDg2MWY2Mjc5MmYwNzNlM2RiMmY2ZjQxOTRhN2I3NzA2In0.nKnLeLpVv8qmeobQSB-4OyuXYQ27OKFDqyjJUdY5vjlLgikKpdqbrIxEOuVYPG49FsiBCjkfiGxuWt0GcnEBKSz91CgY6nG6SC9QS1XVASGb32SyfbgimG3exB92ze8jSQsnAgQPA84m64jsldCwYdvZFkSGRoZM1YHeZSl6e9I'
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


def execute_query_by_id(org, domain, schema, version, query_id):
    url = "https://kg.humanbrainproject.eu/query/{}/{}/{}/{}/{}/instances?databaseScope=RELEASED".format(
        org,
        domain,
        schema,
        version,
        query_id)
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

