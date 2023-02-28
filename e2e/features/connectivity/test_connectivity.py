import siibra
import pytest
from typing import List

from siibra.features.connectivity.regional_connectivity import RegionalConnectivity
from e2e.util import check_duplicate

jba_29 = siibra.parcellations["2.9"]

def test_id_unique():
    features = siibra.features.get(jba_29, RegionalConnectivity)
    duplicates = check_duplicate([f.id for f in features])
    assert len(duplicates) == 0

def test_feature_unique():
    features = siibra.features.get(jba_29, RegionalConnectivity)
    duplicates = check_duplicate([f for f in features])
    assert len(duplicates) == 0

# the get_matrix is a rather expensive operation
# only do once for the master list
def test_connectivity_get_matrix():
    features: List['RegionalConnectivity'] = siibra.features.get(jba_29, "connectivity")
    
    for f in features:
        assert isinstance(f, RegionalConnectivity)
        assert len(f.subjects) > 0
        assert all(isinstance(subject, str) for subject in f.subjects)
        for subject in f.subjects:
            f.get_matrix(subject)


args = [
    (jba_29, "StreamlineCounts"),
    (jba_29, "RegionalConnectivity"),
    (jba_29, RegionalConnectivity),
    (jba_29, "connectivity"),
    (jba_29, siibra.features.connectivity.StreamlineCounts),
    (jba_29, siibra.features.connectivity.StreamlineLengths),
]

@pytest.mark.parametrize("concept,query_arg", args)
def test_get_connectivity(concept, query_arg):
    features: List['RegionalConnectivity'] = siibra.features.get(concept, query_arg)
    assert len(features) > 0, f"Expecting some features exist, but none exist."

