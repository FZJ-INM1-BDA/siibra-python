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
"""Package handling variety of volumes and volume operations"""

from .parcellationmap import Map
from .providers import provider
from .volume import from_array, from_file, from_pointset, from_nifti, Volume

from ..commons import logger
from typing import List, Union
import numpy as np


def warm_cache():
    """Preload preconfigured parcellation maps."""
    _ = Map.registry()
