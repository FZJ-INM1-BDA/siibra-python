
from siibra import get_region
from siibra.features_beta import get

reg = get_region("julich brain 2.9", "hoc1")
features = get(reg, "receptor profile")
data = [datum for f in features for datum in f.get_data()]
assert len(data) > 0, f"expected at least 1 data"
