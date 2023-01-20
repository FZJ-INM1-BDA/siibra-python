from siibra.features.molecular.receptor_density_fingerprint import ReceptorDensityFingerprint
import pytest

all_features = ReceptorDensityFingerprint.get_instances()

@pytest.mark.parametrize('feat', all_features)
def test_receptor_fp(feat: ReceptorDensityFingerprint):
    assert isinstance(feat, ReceptorDensityFingerprint)
    print(feat.name, feat.id)
    print(feat.unit)
    print(feat.neurotransmitters)
    print(feat.data)
