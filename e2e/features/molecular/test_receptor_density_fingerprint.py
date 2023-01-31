from siibra.features.molecular.receptor_density_fingerprint import ReceptorDensityFingerprint
import pytest
from e2e.util import check_duplicate

all_features = ReceptorDensityFingerprint.get_instances()

def test_dup_id():
    dup = check_duplicate([f.id for f in all_features])
    assert len(dup) == 0, f"Expecting no duplicated ids, but got duplicated ids: {', '.join(list(dup))}"

@pytest.mark.parametrize('feat', all_features)
def test_receptor_fp(feat: ReceptorDensityFingerprint):
    assert isinstance(feat, ReceptorDensityFingerprint)
    print(feat.name, feat.id)
    print(feat.unit)
    print(feat.neurotransmitters)
    print(feat.data)
