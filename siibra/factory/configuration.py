import json
from collections import defaultdict
from typing import Dict, List

from .factory import build_feature, build_space, build_parcellation, build_map
from .iterator import attribute_collection_iterator
from ..atlases import Space, Parcellation, Region, parcellationmap
from ..descriptions import register_modalities, Modality, RegionSpec
from ..concepts import QueryParamCollection, Feature
from ..assignment.assignment import filter_by_query_param
from ..retrieval_new.file_fetcher import (
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


@register_modalities()
def register_cell_body_density():
    yield Modality(value="Cell body density")


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
    from ..descriptions import RegionSpec

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


@filter_by_query_param.register(Feature)
def iter_cell_body_density(input: QueryParamCollection):
    mods = [mod for cri in input.criteria for mod in cri._find(Modality)]
    if Modality(value="Cell body density") not in mods:
        return

    from ..concepts import QueryParam
    from ..descriptions import RegionSpec, ID, Name

    name_to_regionspec: Dict[str, RegionSpec] = {}
    returned_features: Dict[str, List[Feature]] = defaultdict(list)

    for feature in iter_preconf_features(
        QueryParam(attributes=[Modality(value="Segmented cell body density")])
    ):
        try:
            regionspec = feature._get(RegionSpec)
            returned_features[regionspec.value].append(feature)
            name_to_regionspec[regionspec.value] = regionspec
        except Exception as e:
            logger.warn(f"Processing {feature} resulted in exception {str(e)}")

    for regionname, features in returned_features.items():
        yield Feature(
            attributes=[
                *[
                    attr
                    for feature in features
                    for attr in feature.attributes
                    if not isinstance(attr, (RegionSpec, Name, ID))
                ],
                name_to_regionspec[regionname],
            ]
        )
