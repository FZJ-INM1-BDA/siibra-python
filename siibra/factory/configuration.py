from .factory import build_object, build_feature
from ..concepts.feature import Feature
from ..concepts.attribute_collection import AttributeCollection
from ..configuration.configuration import Configuration
from ..assignment.assignment import register_collection_generator, match

@register_collection_generator(Feature)
def iter_preconf_features(filter_param: AttributeCollection):
    cfg = Configuration()

    # below should produce the same result
    # all_features = [build_object(s) for _, s in cfg.specs.get("siibra/feature/v0.2")]
    all_features = [build_feature(s) for _, s in cfg.specs.get("siibra/feature/v0.2")]
    for feature in all_features:
        if match(filter_param, feature):
            yield feature

