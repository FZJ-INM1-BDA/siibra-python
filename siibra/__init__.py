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

from .commons import logger, QUIET, VERBOSE, MapType, MapIndex, set_log_level, __version__
from .core import Atlas, Parcellation, Space
from .locations import Point, PointSet, BoundingBox
from .retrieval import EbrainsRequest, CACHE
from .configuration import Configuration
from .features import Feature

import os


logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)

# forward access to some functions
set_ebrains_token = EbrainsRequest.set_token
fetch_ebrains_token = EbrainsRequest.fetch_token
clear_cache = CACHE.clear
get_features = Feature.match
use_configuration = Configuration.use_configuration
extend_configuration = Configuration.extend_configuration
modalities = Feature.modalities
find_regions = Parcellation.find_regions


# lazy access to class registries
# (should only be executed on request, not on package intialization)
def __getattr__(attr: str):

    aliases = {
        'atlases': Atlas,
        'spaces': Space,
        'parcellations': Parcellation,
    }

    # provide siibra.atlases, siibra.spaces, ...
    if attr in aliases:
        return aliases[attr].registry()

    # provide siibra.get_atlas(..), ...
    if attr.startswith('get_'):
        name = attr.split('_')[1].capitalize()
        for cls in aliases.values():
            if cls.__name__ == name:
                return cls.get_instance

    raise AttributeError(f"siibra has no attribute named {attr}")


def get_template(space_spec: str, **kwargs):
    return (
        Space
        .get_instance(space_spec)
        .get_template(**kwargs)
    )

def get_map(parc_spec: str, space_spec: str, maptype: MapType = MapType.LABELLED):
    return (
        Parcellation
        .get_instance(parc_spec)
        .get_map(space=space_spec, maptype=maptype)
    )


def get_region(parc_spec: str, region_spec: str):
    return (
        Parcellation
        .get_instance(parc_spec)
        .get_region(regionspec=region_spec)
    )


def set_feasible_download_size(maxsize_gbyte):
    from .volumes import volume
    volume.gbyte_feasible = maxsize_gbyte
    logger.info(f"Set feasible download size to {maxsize_gbyte} GiB.")


def set_cache_size(maxsize_gbyte: int):
    assert maxsize_gbyte >= 0
    CACHE.SIZE_GIB = maxsize_gbyte
    logger.info(f"Set cache size to {maxsize_gbyte} GiB.")


if "SIIBRA_CACHE_SIZE_GIB" in os.environ:
    set_cache_size(float(os.environ.get("SIIBRA_CACHE_SIZE_GIB")))
