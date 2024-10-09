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


from typing import Union, Type, List, TypeVar
import os as _os

from .exceptions import NotFoundException
from .commons.logger import logger, QUIET, VERBOSE, set_log_level
from .commons.iterable import assert_ooo
from .commons.instance_table import LazyBkwdCompatInstanceTable
from .commons.tree import collapse_nodes
from .cache import fn_call_cache, Warmup, WarmupLevel, CACHE as cache

from . import factory
from . import operations
from . import attributes
from ._version import __version__
from .atlases import Space, ParcellationScheme, Region, parcellationmap
from .atlases.region import filter_newest
from .attributes import Attribute, AttributeCollection
from .attributes.descriptions import Modality, RegionSpec, Gene
from .attributes.descriptions.modality import modality_vocab
from .attributes.locations import Location, Point, PointCloud, BoundingBox
from .attributes.datarecipes import DataRecipe
from .concepts import AtlasElement, QueryParam, Feature
from .assignment import (
    SearchResult,
    preprocess_concept,
)
from .factory.configuration import iter_preconfigured

logger.info(f"Version: {__version__}")
logger.warning("This is a development release. Use at your own risk.")
logger.info(
    "Please file bugs and issues at https://github.com/FZJ-INM1-BDA/siibra-python."
)

T = TypeVar("T", bound=AttributeCollection)


def find(criteria: List[AttributeCollection], find_type: Type[T]):
    res = SearchResult(criteria=criteria, search_type=find_type)
    return res.find()


def get_space(space_spec: str):
    """Convenient access to reference space templates."""
    return assert_ooo(find_spaces(space_spec))


@fn_call_cache
def find_spaces(space_spec: str):
    critiera = SearchResult.str_search_criteria(space_spec)
    searchresult = SearchResult(criteria=critiera, search_type=Space)
    return searchresult.find()


def fetch_template(
    space_spec: str, frmt: str = None, variant: str = None, fragment: str = ""
):
    return get_space(space_spec).get_template(frmt=frmt, variant=variant)


def get_parcellation(parc_spec: str):
    """Convenient access to parcellations."""
    searched_parcs = find_parcellations(parc_spec)
    newest_versions = [
        p
        for p in searched_parcs
        if p.is_newest_version or p.next_version not in searched_parcs
    ]
    return assert_ooo(newest_versions)


@fn_call_cache
def find_parcellations(parc_spec: str):
    critiera = SearchResult.str_search_criteria(parc_spec)
    searchresult = SearchResult(criteria=critiera, search_type=ParcellationScheme)
    return searchresult.find()


def find_maps(
    parcellation: Union[ParcellationScheme, str, None] = None,
    space: Union[Space, str, None] = None,
    maptype: Union[None, str] = None,
    name: str = "",
):
    """Convenient access to parcellation maps."""

    from .attributes.descriptions import SpaceSpec, ParcSpec
    from .concepts import QueryParam

    queries = []
    if space:
        if isinstance(space, Space):
            space = space.ID
        space_query = QueryParam(attributes=[SpaceSpec(value=space)])
        queries.append(space_query)
    if parcellation:
        if isinstance(parcellation, ParcellationScheme):
            parcellation = parcellation.ID
        parc_query = QueryParam(attributes=[ParcSpec(value=parcellation)])
        queries.append(parc_query)

    searchresult = SearchResult(criteria=queries, search_type=parcellationmap.Map)
    return [
        mp
        for mp in searchresult.find()
        if (maptype is None or mp.maptype == maptype) and name in mp.name
    ]


def get_map(parcellation: str, space: str, maptype: str = "labelled", name: str = ""):
    """Convenient access to parcellation maps."""
    searched_maps = find_maps(parcellation, space, maptype, name)
    return assert_ooo(
        searched_maps,
        lambda maps: (
            (
                "The specification matched multiple maps. Specify one of their ",
                "names as the `name` keyword argument.\n",
                "\n".join(f"- {m.name}" for m in maps),
            )
            if len(maps) > 1
            else """The specification matched no maps."""
        ),
    )


def find_features(
    concept: Union[AtlasElement, Location, DataRecipe],
    modality: Union[Modality, str],
    **kwargs,
):
    concept = preprocess_concept(concept)

    if isinstance(modality, str):
        mod_str = modality
        modality = modality_vocab.modality[modality]
        logger.info(f"Provided {mod_str} parsed as {modality}")

    assert isinstance(
        modality, Modality
    ), f"Expecting modality to be of type str or Modality, but is {type(modality)}."

    modality_query_param = QueryParam(attributes=[modality])

    query_ac = [modality_query_param, concept]

    if "genes" in kwargs:
        assert isinstance(kwargs["genes"], list)
        gene_ac = AttributeCollection(
            attributes=[
                (
                    Gene(gene["symbol"])
                    if isinstance(gene, dict)
                    else Gene(value=gene.upper())
                )
                for gene in kwargs["genes"]
            ]
        )
        query_ac.append(gene_ac)

    if "gene" in kwargs:
        if isinstance(kwargs["gene"], dict):
            gene = kwargs["gene"]["symbol"]
        else:
            gene = kwargs["gene"]
        assert isinstance(gene, str)
        gene_ac = AttributeCollection(attributes=[Gene(value=gene.upper())])
        query_ac.append(gene_ac)

    # place modality_query_param first, since it shortcircuits alot quicker
    return find(
        query_ac,
        Feature,
    )


# convenient access to regions of a parcellation
def get_region(parcellation: str, region: str) -> Region:
    found_regions = find_regions(parcellation, region)
    if len(found_regions) == 0:
        raise NotFoundException(
            f"parcellation_spec={parcellation!r} and regionspec={region!r} found no regions"
        )
    exact_match = [r for r in found_regions if r.name == region]
    if len(exact_match) > 0:
        logger.debug(
            f"{len(exact_match)} exact match for {region} found. Returning first exact match."
        )
        return exact_match[0]

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
            [
                QueryParam(
                    attributes=[
                        RegionSpec(parcellation_id=parcellation_id, value=regionspec)
                    ]
                )
            ],
            Region,
        )
    ]


spaces = LazyBkwdCompatInstanceTable(
    getitem=get_space,
    get_elements=lambda: {spc.name: spc for spc in iter_preconfigured(Space)},
)

parcellations = LazyBkwdCompatInstanceTable(
    getitem=get_parcellation,
    get_elements=lambda: {
        spc.name: spc for spc in iter_preconfigured(ParcellationScheme)
    },
)


def _not_implemented(*args):
    raise NotImplementedError("map getitem not yet implemented")


maps = LazyBkwdCompatInstanceTable(
    getitem=_not_implemented,
    get_elements=lambda: {
        mp.name: mp for mp in iter_preconfigured(parcellationmap.Map)
    },
)


def set_feasible_download_size(maxsize_gbyte):
    from .operations import volume_fetcher

    volume_fetcher.SIIBRA_MAX_FETCH_SIZE_GIB = maxsize_gbyte
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
