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
from siibra.openminds.base import SiibraBaseModel

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
openminds_ref = '3fa86f956b407b2debf47c2e1b6314e37579c707'
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
        output=Path(output_filename),
        disable_timestamp=True,
        base_class="siibra.openminds.base.SiibraBaseModel"
    )
    with open(output_filename, "r") as fp:
        txt = fp.read()
    with open(output_filename, "w") as fp:
        fp.write(txt.replace("https://openminds.ebrains.eu/vocab/", ""))


def add_init():
    def process_dir(dirpath: str):

        # ignore directories starts with _
        if dirpath[:1] == "_":
            return

        # ignore if path is not directory
        if not os.path.isdir(dirpath):
            return
        
        # check if __init__.py exists. if not, create it.
        init_path = os.path.join(dirpath, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w"):
                pass
        
        # iterate all existing items in directory
        for fname in os.listdir(dirpath):
            process_dir(path.join(dirpath, fname))
    
    for fname in os.listdir(path_to_currdir):
        process_dir(path.join(path_to_currdir, fname))
    

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
    add_init()
    

if __name__ == "__main__":
    main()