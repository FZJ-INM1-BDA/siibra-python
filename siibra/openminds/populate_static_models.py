"""
n.b.

this script requires datamodel-code-generator, which is not installed via requirements.
"""


from os import path
from pathlib import Path
import os
from collections import namedtuple
from typing import List
import requests
from datamodel_code_generator import generate, InputFileType
from datamodel_code_generator.format import PythonVersion

path_to_currdir = path.dirname(
    path.abspath(__file__)
)
raw_url = 'https://raw.githubusercontent.com/HumanBrainProject/openMINDS/{ref}/{openminds_v}/{domain}/{schema}.schema.json'
"""
domain typically follows
[a-zA-Z]+/v[0-9]+

e.g. SANDS/v3


schema typeicall follows
[a-zA-Z]+/[a-zA-Z]+

e.g. miscellaneous/coordinatePoint
"""

OpenmindsSchema = namedtuple('OpenmindsSchema', ['domain', 'schema'])
openminds_ref = 'dbb4c54'
openminds_v = 'v3'

def process_schema(model: OpenmindsSchema):
    url = raw_url.format(
        ref=openminds_ref,
        openminds_v=openminds_v,
        domain=model.domain,
        schema=model.schema,
    )
    resp = requests.get(url)
    assert resp.status_code == 200
    resp_text = resp.text

    target_dir = f'{path_to_currdir}/{model.domain}/{path.dirname(model.schema)}'
    os.makedirs(target_dir, exist_ok=True)

    output_filename = f'{path_to_currdir}/{model.domain}/{model.schema}.py'
    generate(
        resp_text,
        target_python_version=PythonVersion.PY_36,
        input_file_type=InputFileType.JsonSchema,
        input_filename=url,
        snake_case_field=True,
        output=Path(output_filename)
    )


def main():
    resp = requests.get(f'https://api.github.com/repos/HumanBrainProject/openMINDS/git/trees/{openminds_ref}?recursive=1')
    assert resp.status_code == 200
    resp_json = resp.json()
    tree = resp_json.get('tree')
    import re
    starts_with = re.compile(f'^{openminds_v}')
    ends_with = re.compile(r'\.schema\.json$')
    filtered_tree: List[str] = [t.get('path') for t in tree
        if starts_with.search(t.get('path'))
        and ends_with.search(t.get('path'))
        and t.get('type') == 'blob'    
    ]
    find_domain_schema = re.compile(f'^{openminds_v}/([a-zA-Z]+\/v[0-9]+)\/(.+)\.schema\.json$')
    
    
    for filepath in filtered_tree:
        full_str = find_domain_schema.search(filepath)
        domain, schema = full_str.groups()
        
        process_schema(
            OpenmindsSchema(domain, schema)
        )
    

if __name__ == "__main__":
    main()