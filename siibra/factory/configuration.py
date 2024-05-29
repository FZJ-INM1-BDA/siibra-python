import json

from .factory import build_feature, build_space, build_parcellation
from ..atlases import Space, Parcellation
from ..descriptions import register_modalities, Modality
from ..concepts.feature import Feature
from ..concepts.attribute_collection import AttributeCollection
from ..assignment.assignment import register_collection_generator, match
from ..retrieval_new.file_fetcher import (
    GithubRepository,
    LocalDirectoryRepository,
)
from ..commons import SIIBRA_USE_CONFIGURATION, logger
from ..exceptions import UnregisteredAttrCompException


class Configuration:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if SIIBRA_USE_CONFIGURATION:
            logger.warning(
                "config.SIIBRA_USE_CONFIGURATION defined, use configuration"
                f"at {SIIBRA_USE_CONFIGURATION}"
            )
            self.default_repos = [
                LocalDirectoryRepository.from_url(SIIBRA_USE_CONFIGURATION)
            ]
        else:
            self.default_repos = [
                GithubRepository(
                    "FZJ-INM1-BDA",
                    "siibra-configurations",
                    reftag="refactor_attr",
                    eager=True,
                )
            ]

    def iter_jsons(self, prefix: str):
        repo = self.default_repos[0]

        for file in repo.search_files(prefix):
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


def _iter_preconf_spaces():
    # TODO replace/migrate old configuration here
    cfg = Configuration()

    # below should produce the same result
    return [build_space(obj) for obj in cfg.iter_jsons("spaces")]


@register_collection_generator(Space)
def iter_preconf_spaces(filter_param: AttributeCollection):
    for space in _iter_preconf_spaces():
        try:
            if match(filter_param, space):
                yield space
        except UnregisteredAttrCompException:
            continue


def _iter_preconf_parcellations():
    # TODO replace/migrate old configuration here
    cfg = Configuration()
    return [build_parcellation(obj) for obj in cfg.iter_jsons("parcellations")]


@register_collection_generator(Parcellation)
def iter_preconf_parcellations(filter_param: AttributeCollection):
    for parcellation in _iter_preconf_parcellations():
        try:
            if match(filter_param, parcellation):
                yield parcellation
        except UnregisteredAttrCompException:
            continue


def _iter_preconf_features():
    cfg = Configuration()

    # below should produce the same result
    # all_features = [build_object(s) for _, s in cfg.specs.get("siibra/feature/v0.2")]
    return [build_feature(obj) for obj in cfg.iter_type("siibra/concepts/feature/v0.2")]


@register_collection_generator(Feature)
def iter_preconf_features(filter_param: AttributeCollection):
    for feature in _iter_preconf_features():
        try:
            if match(filter_param, feature):
                yield feature
        except UnregisteredAttrCompException:
            continue


@register_modalities()
def iter_modalities():
    for feature in _iter_preconf_features():
        yield from feature.getiter(Modality)
