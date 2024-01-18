import siibra
import pytest
from siibra.features.image.image import Image

# Update this as new configs are added
results = [
    (siibra.features.get(siibra.get_template("big brain"), "CellbodyStainedSection"), 145),
    (siibra.features.get(siibra.get_template("big brain"), "CellBodyStainedVolumeOfInterest"), 2),
    (siibra.features.get(siibra.get_template("mni152"), "image", restrict_space=True), 4),
    (siibra.features.get(siibra.get_template("mni152"), "image", restrict_space=False), 13),
    (siibra.features.get(siibra.get_region('julich 3', 'hoc1 left'), "CellbodyStainedSection"), 47),
    (siibra.features.get(siibra.get_region('julich 2.9', 'hoc1 left'), "CellbodyStainedSection"), 41)
]
features = [f for fts, _ in results for f in fts]


@pytest.mark.parametrize("feature", features)
def test_feature_has_datasets(feature: Image):
    assert len(feature.datasets) > 0


@pytest.mark.parametrize("features, result_len", results)
def test_image_query_results(
    features: Image,
    result_len: int
):
    assert len(features) == result_len
