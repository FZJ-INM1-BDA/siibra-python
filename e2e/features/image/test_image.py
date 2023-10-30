import siibra
import pytest

features = siibra.features.get(
    siibra.get_template("big brain"), "CellBodyStainedVolumeOfInterest"
)


@pytest.mark.parametrize("feature", features)
def test_feature_has_datasets(
    feature: siibra.features.image.CellBodyStainedVolumeOfInterest,
):
    assert len(feature.datasets) > 0
