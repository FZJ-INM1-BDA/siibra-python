from typing import Type

import siibra
from siibra.concepts.attribute_collection import AttributeCollection
from siibra.concepts.feature import Feature
from siibra.assignment.assignment import get, match
from siibra.descriptions import Modality, RegionSpec
from siibra.concepts.query_parameter import QueryParam
import siibra.descriptions

region_spec = RegionSpec(value="Area hOc1 (V1, 17, CalcS)")
modality = Modality(value="Neurotransmitter receptor density fingerprint")

# query by regionspec... works?
query1 = QueryParam(attributes=[region_spec])
features1 = list(get(query1, Feature))
print(len(features1))

# query by modality works
query2 = QueryParam(attributes=[modality])
features2 = list(get(query2, Feature))
print(len(features2))

# but combining them does not work
# since the check goes by any check
query3 = QueryParam(attributes=[
    region_spec,
    modality,
])
features3 = list(get(query3, Feature))
print(len(features3))


# I suspect we will have to introduce helper methods like below to achieve `and` check
def feature_get(qp: QueryParam, col_type: Type[AttributeCollection]):
    for feat in get(qp, col_type):
        if all(
            match(new_qp, feat)
            for new_qp in qp.split_attrs()
        ):
            yield feat

features4 = list(feature_get(query3, Feature))
print(len(features4))


# testing modality autocomplete
# works at runtime (not statically)
print(
    "testing modality dir",
    siibra.descriptions.modality.vocab.__dir__()
)

# # testing querying via bounding box
# bbox = BoundingBox(
#     (-11, -11, -11),
#     (11, 11, 11),
#     "big brain"
# )
# features = get(bbox, "cell body staining")
# assert len(features) > 0



# # segmented cell body density
# reg = get_region("julich brain 2.9", "hoc1 left")
# features = get(reg, modality.SEGMENTED_CELL_BODY_DENSITY)
# feat = features[0]
# data, = list(feat.get_data(feat.data_modalities.DENSITY_IMAGE)) # autocomplete works at runtime

# assert data is not None

# # cell body density
# reg = get_region("julich brain 2.9", "hoc1")
# features = get(reg, modality.CELL_BODY_DENSITY)
# assert len(features) > 1
# filtered_features = [feature for feature in features if any(mod == modality.CELL_BODY_DENSITY for mod in feature.get_modalities())]
# feat, = filtered_features

# data = [datum for datum in feat.get_data("layer statistics")]
# assert all(datum is not None for datum in data)

