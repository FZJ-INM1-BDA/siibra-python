
from siibra import get_region
from siibra.locations import BoundingBox
from siibra.features_beta import get, modality
from siibra.features_beta.attributes.meta_attributes import ModalityAttribute

reg = get_region("julich brain 2.9", "hoc1")
features = get(reg, "receptor fingerprint")
f = features[0]
data = list(f.get_data())[0]
data = [datum for f in features for datum in f.get_data()]
assert len(data) > 0, f"expected at least 1 data"

bbox = BoundingBox(
    (-11, -11, -11),
    (11, 11, 11),
    "big brain"
)
features = get(bbox, "cell body staining")

assert len(features) > 0

print(
    modality.__dir__()
)
reg = get_region("julich brain 2.9", "hoc1 left")
features = get(reg, modality.SEGMENTED_CELL_BODY_DENSITY)
feat = features[0]
data, = list(feat.get_data(feat.data_modalities.DENSITY_IMAGE))

assert data is not None