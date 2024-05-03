
from siibra import get_region
from siibra.locations import BoundingBox
from siibra.features_beta import get, modality
from siibra.features_beta.feature import get_layer_statistics

# testing receptor fingerprint
reg = get_region("julich brain 2.9", "hoc1")
features = get(reg, "receptor fingerprint")
f = features[0]
data = list(f.get_data())[0]
data = [datum for f in features for datum in f.get_data()]
assert len(data) > 0, f"expected at least 1 data"

# testing querying via bounding box
bbox = BoundingBox(
    (-11, -11, -11),
    (11, 11, 11),
    "big brain"
)
features = get(bbox, "cell body staining")
assert len(features) > 0


# testing modality autocomplete
# works at runtime (not statically)
print(
    "testing modality dir",
    modality.__dir__()
)

# segmented cell body density
reg = get_region("julich brain 2.9", "hoc1 left")
features = get(reg, modality.SEGMENTED_CELL_BODY_DENSITY)
feat = features[0]
data, = list(feat.get_data(feat.data_modalities.DENSITY_IMAGE)) # autocomplete works at runtime

assert data is not None

# cell body density
reg = get_region("julich brain 2.9", "hoc1")
features = get(reg, modality.CELL_BODY_DENSITY)
assert len(features) > 1
filtered_features = [feature for feature in features if any(mod == modality.CELL_BODY_DENSITY for mod in feature.get_modalities())]
feat, = filtered_features

data = [datum for datum in feat.get_data("layer statistics")]
assert all(datum is not None for datum in data)

