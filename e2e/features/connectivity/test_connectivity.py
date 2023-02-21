import siibra
import pytest
from typing import List

from siibra.features.connectivity.regional_connectivity import RegionalConnectivity
from e2e.util import check_duplicate

jba_29 = siibra.parcellations["2.9"]


# FIXME
@pytest.mark.xfail(reason="Some regional connectivity have same id. Remove mark as xfail when fixed.")
def test_id_unique():
    features = siibra.features.get(jba_29, RegionalConnectivity)
    duplicates = check_duplicate([f.id for f in features])
    assert len(duplicates) == 0


args = [
    (jba_29, "StreamlineCounts"),
    (jba_29, "RegionalConnectivity"),
    (jba_29, RegionalConnectivity),
    (jba_29, "connectivity"),
    (jba_29, siibra.features.connectivity.StreamlineCounts),
    (jba_29, siibra.features.connectivity.StreamlineLengths),
]


@pytest.mark.parametrize("concept,query_arg", args)
def test_connectivity(concept, query_arg):
    features: List['RegionalConnectivity'] = siibra.features.get(concept, query_arg)
    assert len(features) > 0

    for f in features:
        assert isinstance(f, RegionalConnectivity)
        assert len(f.subjects) > 0
        assert all(isinstance(subject, str) for subject in f.subjects)
        for subject in f.subjects:
            f.get_matrix(subject)
