import siibra
import pytest
from typing import List

from siibra.features.connectivity.regional_connectivity import RegionalConnectivity
from e2e.util import check_duplicate

features = [f
        for Cls in siibra.features.feature.Feature.SUBCLASSES[RegionalConnectivity]
        for f in Cls.get_instances()]

def test_id_unique():
    duplicates = check_duplicate([f.id for f in features])
    assert len(duplicates) == 0

def test_feature_unique():
    duplicates = check_duplicate([f for f in features])
    assert len(duplicates) == 0

@pytest.mark.parametrize('f', features)
def test_connectivity_get_matrix(f: RegionalConnectivity):
    assert isinstance(f, RegionalConnectivity)
    assert len(f.subjects) > 0
    assert all(isinstance(subject, str) for subject in f.subjects)
    matrix_df = f.get_matrix()
    assert all(matrix_df.index[i] == r for i, r in enumerate(matrix_df.columns))
    for subject in f.subjects:
        matrix_df = f.get_matrix(subject)
        assert all(
            matrix_df.index[i] == r for i, r in enumerate(matrix_df.columns)
        )

jba_29 = siibra.parcellations['2.9']

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

def test_copy_is_returned():
    feat:RegionalConnectivity = features[0]

    # retrieve matrix
    matrix = feat.get_matrix(feat.subjects[0])

    # ensure new val to be put is different from prev val
    prev_val = matrix.iloc[0, 0]
    new_val = 42
    assert new_val != prev_val
    matrix.iloc[0, 0] = new_val

    # retrieve matrix again
    matrix_again = feat.get_matrix(feat.subjects[0])
    assert matrix_again.iloc[0, 0] == prev_val
