# Copyright 2018-2024
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

from dataclasses import dataclass, field
from typing import Dict

from ..base import Data
from ....retrieval.volume_fetcher import IMAGE_FORMATS, MESH_FORMATS, Mapping


FORMAT_LOOKUP = {
    None: IMAGE_FORMATS + MESH_FORMATS,
    "mesh": MESH_FORMATS,
    "image": IMAGE_FORMATS,
}


@dataclass
class Volume(Data):
    schema: str = "siibra/attr/data/volume"
    space_id: str = None
    format: str = None
    url: str = None
    mapping: Dict[str, Mapping] = field(default=None, repr=False)
    colormap: str = field(default=None, repr=False)