from siibra.features.tabular.receptor_density_fingerprint import (
    ReceptorDensityFingerprint,
)
import pytest
from e2e.util import check_duplicate

all_features = ReceptorDensityFingerprint._get_instances()


def test_dup_id():
    dup = check_duplicate([f for f in all_features], lambda f: f.id)
    assert (
        len(dup) == 0
    ), "Expecting no duplicated ids, but got duplicated ids:" + "\n".join(
        [f.name + " " + f.id for f in list(dup)]
    )


@pytest.mark.parametrize("feat", all_features)
def test_receptor_fp(feat: ReceptorDensityFingerprint):
    assert isinstance(feat, ReceptorDensityFingerprint)
    print(feat.name, feat.id)
    print(feat.unit)
    print(feat.neurotransmitters)
    print(feat.data)
