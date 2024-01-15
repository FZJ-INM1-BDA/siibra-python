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
    __version__
)

from .core import (
    atlas as _atlas,
    parcellation as _parcellation,
    space as _space
)
from .volumes import parcellationmap as _parcellationmap
from .retrieval.requests import (
    EbrainsRequest as _EbrainsRequest,
    CACHE as cache
)
from . import configuration
from .configuration import factory
from . import features, livequeries
from siibra.locations import Point, PointSet

import os as _os
logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)

# forward access to some functions
set_ebrains_token = _EbrainsRequest.set_token
fetch_ebrains_token = _EbrainsRequest.fetch_token
find_regions = _parcellation.Parcellation.find_regions
from_json = factory.Factory.from_json


def __getattr__(attr: str):
    # lazy loading of some classes for package-level functions.
    if attr == 'atlases':
        return _atlas.Atlas.registry()
    elif attr == 'spaces':
        return _space.Space.registry()
    elif attr == 'parcellations':
        return _parcellation.Parcellation.registry()
    elif attr == 'maps':
        return _parcellationmap.Map.registry()
    elif attr == 'use_configuration':
        return configuration.Configuration.use_configuration
    elif attr == 'extend_configuration':
        return configuration.Configuration.extend_configuration
    else:
        raise AttributeError(f"No such attribute: {__name__}.{attr}")


# convenient access to reference space templates
def get_template(space_spec: str, **kwargs):
    return (
        _space.Space
        .get_instance(space_spec)
        .get_template(**kwargs)
    )


# convenient access to parcellation maps
def get_map(parcellation: str, space: str, maptype: MapType = MapType.LABELLED, **kwargs):
    return (
        _parcellation.Parcellation
        .get_instance(parcellation)
        .get_map(space=space, maptype=maptype, **kwargs)
    )


# convenient access to regions of a parcellation
def get_region(parcellation: str, region: str):
    return (
        _parcellation.Parcellation
        .get_instance(parcellation)
        .get_region(region)
    )


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


def warm_cache():
    """
    Preload preconfigured siibra concepts.

    Siibra relies on preconfigurations that simplify integrating various
    concepts such as parcellations, refernce spaces, and multimodal data
    features. By preloading the instances, siibra commits all preconfigurations
    to the memory at once instead of commiting them when required.
    """
    _ = _atlas.Atlas.registry()
    _ = _space.Space.registry()
    _ = _parcellation.Parcellation.registry()
    _ = _parcellationmap.Map.registry()
    features.warm_cache()
    livequeries.warm_cache()


def __dir__():
    return [
        "atlases",
        "spaces",
        "parcellations",
        "features",
        "use_configuration",
        "extend_configuration",
        "get_region",
        "find_regions",
        "get_map",
        "get_template",
        "MapType",
        "Point",
        "PointSet",
        "QUIET",
        "VERBOSE",
        "fetch_ebrains_token",
        "set_ebrains_token",
        "vocabularies",
        "__version__",
        "cache",
        "warm_cache"
    ]
