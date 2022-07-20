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

from .commons import logger, QUIET, VERBOSE

# __version__ is parsed by setup.py
__version__ = "0.3a24"
logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)

from .core import spaces, parcellations, atlases
from .features import modalities, gene_names, get_features
from .commons import MapType, ParcellationIndex
from .retrieval import EbrainsRequest, CACHE
from .core import Point, PointSet, BoundingBox
from .core.space import Location as _
from .core.region import THRESHOLD_CONTINUOUS_MAPS
from . import samplers
from os import environ

from_sands = _.from_sands
set_ebrains_token = EbrainsRequest.set_token
fetch_ebrains_token = EbrainsRequest.fetch_token
clear_cache = CACHE.clear


def set_feasible_download_size(maxsize_gbyte):
    from .volumes import volume
    volume.gbyte_feasible = maxsize_gbyte
    logger.info(f"Set feasible download size to {maxsize_gbyte} GiB.")


def set_cache_size(maxsize_gbyte: int):
    assert maxsize_gbyte >= 0
    CACHE.SIZE_GIB = maxsize_gbyte
    logger.info(f"Set cache size to {maxsize_gbyte} GiB.")


if "SIIBRA_CACHE_SIZE_GIB" in environ:
    set_cache_size(float(environ.get("SIIBRA_CACHE_SIZE_GIB")))
