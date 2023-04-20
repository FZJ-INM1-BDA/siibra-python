from typing import Type
import pytest
from siibra.features.query import FeatureQuery
from siibra.features.ebrains import EbrainsRegionalFeatureQuery
from siibra.core.serializable_concept import JSONSerializable

# applicable_queries = [
#     pytest.param(query, marks=pytest.mark.xfail(reason="KG down. EbrainsRegionalFeatureQuery gets 500")) if query is EbrainsRegionalFeatureQuery else
#     query
#     for queries in FeatureQuery._implementations.values()
#     for query in queries
#     if issubclass(query._FEATURETYPE, JSONSerializable)]

# @pytest.mark.parametrize('feature_query', applicable_queries)
# def test_all_feat_has_correct_id(feature_query: Type[FeatureQuery]):
#     features = feature_query().features
#     assert len(features) > 0, f"expecting to have at least 1 feature, but got 0"
#     feature = features[0]
#     model = feature.to_model()
#     import re
#     assert re.match(r"^[\w/\-.:]+$", model.id), f"model_id should only contain [\w/\-.:]+, but is instead {model.id}"
#     assert model.id.startswith(model.type)
