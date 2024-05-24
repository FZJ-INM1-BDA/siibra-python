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

from .commons import (
    logger,
    QUIET,
    VERBOSE,
    MapType,
    MapIndex,
    set_log_level,
    __version__,
)

from .commons_new.iterable import assert_ooo

from .cache import Warmup, WarmupLevel, CACHE as cache

from . import factory as factory_new
from . import retrieval_new
from .atlases import Space, Parcellation
from .assignment import string_search
from .exceptions import NotFoundException

import os as _os

logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)


# convenient access to reference space templates
def get_space(space_spec: str):
    searched_spaces = string_search(space_spec, Space)
    return assert_ooo(searched_spaces)


def get_parcellation(parc_spec: str):
    searched_parcs = string_search(parc_spec, Parcellation)
    return assert_ooo(searched_parcs)


# convenient access to parcellation maps
def get_map(
    parcellation: str, space: str, maptype: MapType = MapType.LABELLED, **kwargs
):
    raise NotImplementedError


# convenient access to regions of a parcellation
def get_region(parcellation_spec: str, regionspec: str):
    return get_parcellation(parcellation_spec).get_region(regionspec)


def set_feasible_download_size(maxsize_gbyte):
    from .volumes import volume

    volume.gbyte_feasible = maxsize_gbyte
    logger.info(f"Set feasible download size to {maxsize_gbyte} GiB.")


def set_cache_size(maxsize_gbyte: int):
    assert maxsize_gbyte >= 0
    cache.SIZE_GIB = maxsize_gbyte
    logger.info(f"Set cache size to {maxsize_gbyte} GiB.")


if "SIIBRA_CACHE_SIZE_GIB" in _os.environ:
    set_cache_size(float(_os.environ.get("SIIBRA_CACHE_SIZE_GIB")))


def warm_cache(level=WarmupLevel.INSTANCE):
    """
    Preload preconfigured siibra concepts.

    Siibra relies on preconfigurations that simplify integrating various
    concepts such as parcellations, refernce spaces, and multimodal data
    features. By preloading the instances, siibra commits all preconfigurations
    to the memory at once instead of commiting them when required.
    """
    Warmup.warmup(level)
