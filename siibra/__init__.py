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

from .commons import __version__

from .commons_new.logger import (
    logger,
    QUIET,
    VERBOSE,
    set_log_level
)
from .exceptions import NotFoundException
from .commons_new.string import create_key
from .commons_new.iterable import assert_ooo
from .commons_new.instance_table import InstanceTable
from .commons_new.tree import collapse_nodes

from .cache import Warmup, WarmupLevel, CACHE as cache

from . import factory as factory_new
from . import retrieval_new
from .atlases import Space, Parcellation, Region, parcellationmap
from .atlases.region import filter_newest
from .descriptions import Modality, RegionSpec, Gene
from .descriptions.modality import vocab as modality_types
from .locations import DataClsLocation
from .concepts import AtlasElement, QueryParam, QueryParamCollection, Attribute, AttributeCollection
from .assignment import (
    string_search,
    find,
    collection_match,
    QueryCursor,
)
from .factory.iterator import iter_collection

import os as _os

logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)


spaces = InstanceTable(elements={create_key(spc.name): spc for spc in iter_collection(Space)})
parcellations = InstanceTable(elements={create_key(spc.name): spc for spc in iter_collection(Parcellation)})
maps = InstanceTable(elements={create_key(spc.name): spc for spc in iter_collection(parcellationmap.Map)})


def get_space(space_spec: str):
    """Convenient access to reference space templates."""
    searched_spaces = list(string_search(space_spec, Space))
    return assert_ooo(searched_spaces)


def find_spaces(space_spec: str):
    return list(string_search(space_spec, Space))


def fetch_template(space_spec: str, frmt: str = None, variant: str = "", fragment: str = ""):
    return get_space(space_spec).fetch_template(frmt=frmt, variant=variant)


def get_parcellation(parc_spec: str):
    """Convenient access to parcellations."""
    searched_parcs = list(string_search(parc_spec, Parcellation))
    newest_versions = [p for p in searched_parcs if p.is_newest_version or p.next_version not in searched_parcs]
    return assert_ooo(newest_versions)


def find_parcellations(parc_spec: str):
    return list(string_search(parc_spec, Parcellation))


def find_maps(parcellation: str = None, space: str = None, maptype: str = "labelled", extra_spec: str = ""):
    """Convenient access to parcellation maps."""

    if parcellation:
        try:
            requested_parcellations = find_parcellations(parcellation)
            assert len(requested_parcellations) > 0
        except AssertionError as e:
            raise RuntimeError(f"Requested parcellation {parcellation!r} cannot be found. {str(e)}") from e
    else:
        requested_parcellations = None

    if space:
        try:
            requested_spaces = find_spaces(space)
            assert len(requested_spaces) > 0
        except AssertionError as e:
            raise RuntimeError(f"Requested space {space!r} cannot be found. {str(e)}") from e
    else:
        requested_spaces = None

    return_result = []
    for _map in iter_collection(parcellationmap.Map):
        if _map.maptype != maptype:
            continue
        if requested_parcellations and _map.parcellation not in requested_parcellations:
            continue
        if requested_spaces and _map.space not in requested_spaces:
            continue
        return_result.append(_map)

    return return_result


def get_map(parcellation: str, space: str, maptype: str = "labelled", extra_spec: str = ""):
    """Convenient access to parcellation maps."""
    searched_maps = find_maps(parcellation, space, maptype, extra_spec)
    return assert_ooo(searched_maps)


def find_features(
    concept: Union[AtlasElement, DataClsLocation],
    modality: Union[Modality, str],
    **kwargs,
):
    cursor = get_query_cursor(concept, modality, **kwargs)
    return list(cursor.exec())


def get_query_cursor(
    concept: Union[AtlasElement, DataClsLocation],
    modality: Union[Modality, str],
    **kwargs,
):
    additional_attributes = QueryParam()
    if "genes" in kwargs:
        assert isinstance(kwargs["genes"], list)
        additional_attributes = QueryParam(
            attributes=[Gene(value=gene) for gene in kwargs["genes"]]
        )
    return QueryCursor(
        concept=concept, modality=modality, additional_attributes=additional_attributes
    )

def get_query_collection(
    concept: Union[AtlasElement, Attribute],
    modality: Union[Modality, str],
    **kwargs,
):
    additional_attributes = []
    if "genes" in kwargs:
        assert isinstance(kwargs["genes"], list)
        additional_attributes.append(AttributeCollection(
            attributes=[Gene(value=gene) for gene in kwargs["genes"]]
        ))

    return QueryParamCollection.from_concept_modality(concept, modality,
                                                      additional_attribute_collections=additional_attributes)


# convenient access to regions of a parcellation
def get_region(parcellation_spec: str, regionspec: str):
    found_regions = find_regions(parcellation_spec, regionspec)
    if len(found_regions) == 0:
        raise NotFoundException(
            f"{parcellation_spec=!r} and {regionspec=!r} found no regions"
        )
    if len(found_regions) > 1:
        logger.warning(
            f"Found {len(found_regions)}:\n"
            + "\n".join(f" - {str(r)}" for r in found_regions)
            + "\nSelecting the top most one from the newest versions."
        )
    newest_versions = filter_newest(found_regions)
    return collapse_nodes(newest_versions)[0]


def find_regions(parcellation_spec: str, regionspec: str):
    if parcellation_spec:
        parcellation_ids = [p.ID for p in find_parcellations(parcellation_spec)]
    else:
        parcellation_ids = [None]

    return [
        reg
        for parcellation_id in parcellation_ids
        for reg in find(
            QueryParam(
                attributes=[
                    RegionSpec(parcellation_id=parcellation_id, value=regionspec)
                ]
            ),
            Region,
        )
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
