import os, re, requests, json
from typing import List
path_to_code_snippet='../examples/snippets'

re_is_py=re.compile(r'.py$')
re_is_dunder=re.compile(r'^__')

# TODO walk openapi

open_api_endpoints = [
    'https://siibra-api-jsonable.apps-dev.hbp.eu'
]

class SiibraOpenApi:

    open_api_key = 'x-siibra-code-snippet-id'
    def __init__(self, base_url=None, flavor='expmt', path_to_json='/v1_0/openapi.json'):
        if not base_url:
            raise ValueError('base_url must be defined')
        self.base_url=base_url
        self.flavor = flavor
        self.snippets = {}
        self.server_url = ''

        response = requests.get(f'{base_url}{path_to_json}')
        json_response = json.loads(response.content)
        snippet_dict = {}

        # perhaps something unique to versioned fastapi ?
        self.server_url = json_response.get('servers')[0].get('url')
        for path, path_obj in json_response.get('paths', {}).items():
            if not path_obj.get('get'):
                continue
            get_schema = path_obj.get('get')
            if not get_schema.get(self.open_api_key):
                continue
            snippet_dict[get_schema.get(self.open_api_key)] = {
                'base_url': base_url,
                'operation_id': get_schema.get('operationId'),
            }
        self.snippets = snippet_dict

    
    @property
    def title_md(self):
        return f'[{self.title}]({self.base_url})'

    @property
    def title(self):
        return f'siibra-api [{self.flavor}]'

    def get_snippet_redoc_md(self, filename):
        snippet = self.snippets.get(filename)
        if not snippet:
            return ''
        return f'[redoc]({self.get_snippet_redoc_link(filename)})'
    
    def get_snippet_redoc_link(self, filename):
        snippet = self.snippets.get(filename)
        if not snippet:
            return None
        return f'{self.base_url}{self.server_url}/redoc#operation/{snippet.get("operation_id")}'

def get_github_src(obj):
    dirpath = obj.get('dirpath')
    dirnames = obj.get('dirnames')
    filename = obj.get('filename')
    return f'https://github.com/FZJ-INM1-BDA/siibra-python/blob/feat_jsonableEncoders/code_snippet/{filename}'

def make_row(obj, openapis:List[SiibraOpenApi]=[]):
    dirpath = obj.get('dirpath')
    dirnames = obj.get('dirnames')
    filename = obj.get('filename')
    no_ext = re.sub(r'\.py', '', filename)
    separated = re.sub(r'_', ' ', no_ext)
    return '| {spec} | {github} | {openapi_specs} |'.format(
        spec=separated,
        github=f'[Github]({get_github_src(obj)})',
        openapi_specs=' | '.join([openapi.get_snippet_redoc_md(filename) for openapi in openapis])
    )
    

def gen_code_snippet_md():
    snippets = []
    for dirpath, dirnames, filenames in os.walk(path_to_code_snippet):
        for filename in filenames:
            if re_is_py.search(filename) and not re_is_dunder.search(filename):
                snippets.append({
                    'dirpath': dirpath,
                    'dirnames': dirnames,
                    'filename':  filename
                })
    openapis = [SiibraOpenApi(endpoint) for endpoint in open_api_endpoints]
    return """
# Code snippets

| functionality | siibra-python | {md_table_header} |
| --- | --- | {md_table_sep} |
{rows}
""".format(
    md_table_header=' | '.join([openapi.title_md for openapi in openapis ]),
    md_table_sep=' | '.join([ '---' for _ in openapis]),
    rows='\n'.join([make_row(sn, openapis=openapis) for sn in snippets])
)

if __name__ == '__main__':
    gen_code_snippet_md()