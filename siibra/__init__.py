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

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT_DIR, "VERSION"), "r") as fp:
    __version__ = fp.read()

logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)

from .commons import MapType, MapIndex, set_log_level
from os import environ
from .retrieval.requests import EbrainsRequest
set_ebrains_token = EbrainsRequest.set_token
fetch_ebrains_token = EbrainsRequest.fetch_token
from .retrieval.cache import CACHE
clear_cache = CACHE.clear

from .registry import REGISTRY
use_configuration = REGISTRY.__class__.use_configuration
extend_configuration = REGISTRY.__class__.extend_configuration


def __getattr__(name):
    if name == "atlases":
        return REGISTRY.Atlas
    elif name == "parcellations":
        return REGISTRY.Parcellation
    elif name == "spaces":
        return REGISTRY.Space
    else:
        raise AttributeError(f"No such attribute: {__name__}.{name}")


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

from .features import modalities, get_features
