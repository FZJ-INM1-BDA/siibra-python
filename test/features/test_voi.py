import pytest

from siibra.features.voi import VolumeOfInterestQuery, VolumeOfInterest


query = VolumeOfInterestQuery()

@pytest.mark.parametrize('feature', query.features)
def test_voi_features(feature: VolumeOfInterest):
    feature.to_model()
