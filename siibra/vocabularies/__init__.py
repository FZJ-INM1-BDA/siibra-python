# Copyright 2018-2021
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abbreviations and aliases."""

from ..commons import InstanceTable

import json
from os import path


RT_DIR = path.dirname(__file__)


def runtime_path(fname: str):
    return path.join(RT_DIR, fname)


with open(runtime_path('gene_names.json'), 'r') as f:
    _gene_names = json.load(f)
    GENE_NAMES = InstanceTable[str](
        elements={
            k: {'symbol': k, 'description': v}
            for k, v in _gene_names.items()
        }
    )


with open(runtime_path('receptor_symbols.json'), 'r') as f:
    RECEPTOR_SYMBOLS = json.load(f)


with open(runtime_path('region_aliases.json'), 'r') as f:
    REGION_ALIASES = json.load(f)
