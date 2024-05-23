from .factory import build_object, build_feature, build_space
from ..atlases import Space
from ..descriptions import register_modalities, Modality
from ..concepts.feature import Feature
from ..concepts.attribute_collection import AttributeCollection
from ..configuration.configuration import Configuration
from ..assignment.assignment import register_collection_generator, match

def _iter_preconf_spaces():
    # TODO replace/migrate old configuration here
    cfg = Configuration()

    # below should produce the same result
    return [build_space(s) for _, s in cfg.specs.get("siibra/atlases/space/v0.1")]

@register_collection_generator(Space)
def iter_preconf_spaces(filter_param: AttributeCollection):
    for space in _iter_preconf_spaces():
        if match(filter_param, space):
            yield space


def _iter_preconf_features():
    cfg = Configuration()

    # below should produce the same result
    # all_features = [build_object(s) for _, s in cfg.specs.get("siibra/feature/v0.2")]
    return [build_feature(s) for _, s in cfg.specs.get("siibra/concepts/feature/v0.2")]

@register_collection_generator(Feature)
def iter_preconf_features(filter_param: AttributeCollection):

    for feature in _iter_preconf_features():
        if match(filter_param, feature):
            yield feature

@register_modalities()
def iter_modalities():
    for feature in _iter_preconf_features():
        yield from feature.get(Modality)
