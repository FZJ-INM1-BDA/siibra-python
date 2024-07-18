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

import json
from collections import defaultdict
from typing import Dict, List

from .factory import build_feature, build_space, build_parcellation, build_map
from .iterator import attribute_collection_iterator
from ..atlases import Space, Parcellation, Region, parcellationmap
from ..attributes.descriptions import register_modalities, Modality, RegionSpec
from ..concepts import QueryParamCollection, Feature
from ..assignment.assignment import filter_by_query_param
from ..retrieval.file_fetcher import (
    GithubRepository,
    LocalDirectoryRepository,
)
from ..commons_new.logger import logger
from ..exceptions import UnregisteredAttrCompException

from ..commons import SIIBRA_USE_CONFIGURATION


class Configuration:

    _instance = None

    def __init__(self):
        if SIIBRA_USE_CONFIGURATION:
            logger.warning(
                "config.SIIBRA_USE_CONFIGURATION defined, using configuration "
                f"at {SIIBRA_USE_CONFIGURATION}"
            )
            self.default_repos = [
                LocalDirectoryRepository.from_url(SIIBRA_USE_CONFIGURATION)
            ]
        else:
            repo = GithubRepository("FZJ-INM1-BDA",
                                    "siibra-configurations",
                                    reftag="refactor_attr",
                                    eager=True,)
            self.default_repos = [repo]

    def iter_jsons(self, prefix: str):
        repo = self.default_repos[0]

        for file in repo.search_files(prefix):
            if not file.endswith(".json"):
                logger.debug(f"{file} does not end with .json, skipped")
                continue
            yield json.loads(repo.get(file))

    def iter_type(self, _type: str):
        logger.debug(f"iter_type for {_type}")
        repo = self.default_repos[0]

        for file in repo.search_files():
            if not file.endswith(".json"):
                logger.debug(f"{file} does not end with .json, skipped")
                continue
            try:
                obj = json.loads(repo.get(file))
                if obj.get("@type") == _type:
                    yield obj
            except json.JSONDecodeError:
                continue


#
# register features modalities
#


@register_modalities()
def iter_modalities():
    for feature in _iter_preconf_features():
        yield from feature._finditer(Modality)


#
# Configure how preconfigured AC are fetched and built
#


@attribute_collection_iterator.register(Space)
def _iter_preconf_spaces():
    # TODO replace/migrate old configuration here
    cfg = Configuration()

    # below should produce the same result
    return [build_space(obj) for obj in cfg.iter_jsons("spaces")]


@attribute_collection_iterator.register(Parcellation)
def _iter_preconf_parcellations():
    # TODO replace/migrate old configuration here
    cfg = Configuration()

    return [build_parcellation(obj) for obj in cfg.iter_jsons("parcellations")]


@attribute_collection_iterator.register(Feature)
def _iter_preconf_features():
    cfg = Configuration()

    return [build_feature(obj) for obj in cfg.iter_type("siibra/concepts/feature/v0.2")]


@attribute_collection_iterator.register(parcellationmap.Map)
def _iter_preconf_maps():
    cfg = Configuration()
    return [build_map(obj) for obj in cfg.iter_jsons("maps")]


#
# configure how the preconfigured AC are queried
#


@filter_by_query_param.register(Space)
def register_space(input: QueryParamCollection):
    from .iterator import iter_collection

    for item in iter_collection(Space):
        if input.match(item):
            yield item


@filter_by_query_param.register(Parcellation)
def register_parcellation(input: QueryParamCollection):
    from .iterator import iter_collection

    for item in iter_collection(Parcellation):
        if input.match(item):
            yield item


@filter_by_query_param.register(parcellationmap.Map)
def iter_preconf_parcellationmaps(input: QueryParamCollection):
    from ..attributes.descriptions import RegionSpec

    all_region_specs = [rspec for cri in input.criteria for rspec in cri._find(RegionSpec)]
    input_region_names = [
        region.name
        for regspec in all_region_specs
        for region in regspec.decode()
    ]
    for mp in _iter_preconf_maps():

        try:
            if any(
                (
                    input_region_name in mp.regions
                    for input_region_name in input_region_names
                )
            ):
                yield mp

            if input.match(mp):
                yield mp
        except UnregisteredAttrCompException:
            continue


@filter_by_query_param.register(Feature)
def iter_preconf_features(input: QueryParamCollection):
    for feature in _iter_preconf_features():
        try:
            if input.match(feature):
                yield feature
        except UnregisteredAttrCompException:
            continue


@filter_by_query_param.register(Region)
def iter_region(input: QueryParamCollection):
    from .iterator import iter_collection

    regspecs = [regspec for cri in input.criteria for regspec in cri._find(RegionSpec)]
    if len(regspecs) != 1:
        logger.debug(f"Expected one and only one regionspec, but got {len(regspecs)}")
        return

    regspec = regspecs[0]

    yield from [
        region
        for parc in iter_collection(Parcellation)
        if regspec.parcellation_id is None or regspec.parcellation_id == parc.ID
        for region in parc.find(regspec.value)
    ]

