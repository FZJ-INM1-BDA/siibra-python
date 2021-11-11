from typing import List
import pytest
import siibra
from siibra.core.json_encoder import JSONEncoder
from siibra.features.connectivity import ConnectivityMatrix, ConnectivityProfile
from ..util import get_model
matrix_queries = siibra.features.FeatureQuery.queries('connectivitymatrix')

# assert only one type of receptor query is returned
assert len(matrix_queries) == 1, f'expect exactly 1 (one) return from queries("matrix")'

matrix_query = matrix_queries[0]
matrix_list: List[ConnectivityMatrix] = [rd for rd in matrix_query.features]

@pytest.mark.parametrize('matrix', matrix_list)
def test_all_matrix(matrix: ConnectivityMatrix):
    Model = get_model(ConnectivityMatrix)
    basic_result = JSONEncoder.encode(matrix, nested=True, detail=False)

    Model(**basic_result)
    assert 'matrix' not in basic_result

    full_result = JSONEncoder.encode(matrix, nested=True, detail=True)
    Model(**full_result)
    assert full_result['matrix'].keys()

profile_queries = siibra.features.FeatureQuery.queries('connectivityprofile')
assert len(profile_queries) == 1, f'expect exactly 1 (one) return from queries("connectivityprofile")'

profile_query = profile_queries[0]
profile_list: List[ConnectivityProfile] = [rd for rd in profile_query.features]

@pytest.mark.parametrize('profile', profile_list)
def test_all_profile(profile: ConnectivityProfile):
    Model = get_model(ConnectivityProfile)

    basic_result = JSONEncoder.encode(profile, nested=True, detail=False)
    Model(**basic_result)
    assert 'profile' not in basic_result

    full_result = JSONEncoder.encode(profile, nested=True, detail=True)
    Model(**full_result)
    assert 'profile' in full_result
    assert full_result['profile'].keys()
