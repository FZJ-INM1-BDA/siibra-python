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

from functools import partial
from typing import Union

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
from .commons_new.instance_table import JitInstanceTable

from .cache import Warmup, WarmupLevel, CACHE as cache

from . import factory as factory_new
from . import retrieval_new
from .atlases import Space, Parcellation
from .descriptions import Modality
from .descriptions.modality import vocab as modality_types
from .locations import DataClsLocation
from .concepts import AtlasElement, QueryParam, Feature
from .assignment import (
    string_search,
    attr_col_as_dict,
    iter_attr_col,
    get,
    collection_match,
    QueryCursor,
)

import os as _os

logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)

spaces = JitInstanceTable(getitem=partial(attr_col_as_dict, Space))
parcellations = JitInstanceTable(getitem=partial(attr_col_as_dict, Parcellation))


# convenient access to reference space templates
def get_space(space_spec: str):
    searched_spaces = list(string_search(space_spec, Space))
    return assert_ooo(searched_spaces)


def get_parcellation(parc_spec: str):
    searched_parcs = list(string_search(parc_spec, Parcellation))
    return assert_ooo(searched_parcs)


# convenient access to parcellation maps
def get_map(
    parcellation: str, space: str, maptype: MapType = MapType.LABELLED, **kwargs
):
    raise NotImplementedError


def find_features(
    concept: Union[AtlasElement, DataClsLocation], modality: Union[Modality, str]
):
    cursor = QueryCursor(concept=concept, modality=modality)
    return cursor.exec()


# convenient access to regions of a parcellation
def get_region(parcellation_spec: str, regionspec: str):
    return get_parcellation(parcellation_spec).get_region(regionspec)


def find_regions(regionspec: str):
    return [
        region
        for parc in iter_attr_col(Parcellation)
        for region in parc.find(regionspec)
    ]


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
