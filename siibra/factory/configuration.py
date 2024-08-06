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
from typing import Iterator

from .factory import build_feature, build_space, build_parcellation, build_map
from .iterator import preconfigured_ac_registrar, iter_preconfigured_ac
from .livequery import LiveQuery
from ..atlases import Space, Parcellation, Region, parcellationmap
from ..attributes.descriptions import register_modalities, Modality, RegionSpec
from ..concepts import Feature
from ..retrieval.file_fetcher import (
    GithubRepository,
    LocalDirectoryRepository,
)
from ..commons_new.logger import logger
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
            repo = GithubRepository(
                "FZJ-INM1-BDA",
                "siibra-configurations",
                reftag="refactor_attr",
                eager=True,
            )
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


@preconfigured_ac_registrar.register(Space)
def _iter_preconf_spaces():
    cfg = Configuration()

    # below should produce the same result
    return [build_space(obj) for obj in cfg.iter_jsons("spaces")]


@preconfigured_ac_registrar.register(Parcellation)
def _iter_preconf_parcellations():
    cfg = Configuration()

    return [build_parcellation(obj) for obj in cfg.iter_jsons("parcellations")]


@preconfigured_ac_registrar.register(Feature)
def _iter_preconf_features():
    cfg = Configuration()

    return [build_feature(obj) for obj in cfg.iter_type("siibra/concepts/feature/v0.2")]


@preconfigured_ac_registrar.register(parcellationmap.Map)
def _iter_preconf_maps():
    cfg = Configuration()
    return [build_map(obj) for obj in cfg.iter_jsons("maps")]


class PreconfiguredRegionQuery(LiveQuery[Region], generates=Region):
    """Whilst RegionQuery is not exactly LiveQuery, but presumed performance gain from shortcurcuiting parcellation_id
    should be sufficient reason to use this mechanism to query.

    TODO profile against dumb method
    """

    def generate(self):
        region_specs = [
            attr for attrs in self.find_attributes(RegionSpec) for attr in attrs
        ]

        if len(region_specs) != 1:
            logger.warning(
                f"Region Query cannot deal with total region_specs != 1. You provided {len(region_specs)}"
            )
            return
        regspec = region_specs[0]
        yield from [
            region
            for parc in iter_preconfigured_ac(Parcellation)
            if regspec.parcellation_id is None or regspec.parcellation_id == parc.ID
            for region in parc.find(regspec.value)
        ]


class PreconfiguredMapQuery(
    LiveQuery[parcellationmap.Map], generates=parcellationmap.Map
):
    def generate(self) -> Iterator[parcellationmap.Map]:
        from ..attributes.descriptions import SpaceSpec, ParcSpec, ID, Name
        from ..concepts import QueryParam
        from ..atlases import Space, Parcellation
        from ..assignment import find

        space_specs = [
            spec for specss in self.find_attributes(SpaceSpec) for spec in specss
        ]
        space_query_attrs = [
            attr
            for spec in space_specs
            for attr in (ID(value=spec.value), Name(value=spec.value))
        ]
        spaces = find([QueryParam(attributes=space_query_attrs)], Space)
        if len(spaces) == 0:
            logger.warning(
                f"Cannot find any space with the specification {', '.join([s.value for s in space_specs])}"
            )
            return

        parc_specs = [
            spec for specss in self.find_attributes(ParcSpec) for spec in specss
        ]
        parc_query_attrs = [
            attr
            for spec in parc_specs
            for attr in (ID(value=spec.value), Name(value=spec.value))
        ]
        parcellations = find([QueryParam(attributes=parc_query_attrs)], Parcellation)
        if len(parcellations) == 0:
            logger.warning(
                f"Cannot find any parcellation with the specification {', '.join([s.value for s in parc_specs])}"
            )
            return

        space_ids = [space.ID for space in spaces]
        parc_ids = [parc.ID for parc in parcellations]
        yield from [
            mp
            for mp in iter_preconfigured_ac(parcellationmap.Map)
            if mp.parcellation_id in parc_ids and mp.space_id in space_ids
        ]
