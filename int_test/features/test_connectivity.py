from typing import List
import pytest
import siibra
from siibra.core.json_encoder import JSONEncoder
from siibra.features.connectivity import ConnectivityMatrix

queries = siibra.features.FeatureQuery.queries('matrix')

# assert only one type of receptor query is returned
assert len(queries) == 1

matrix_query = queries[0]
matrix_list: List[ConnectivityMatrix] = [rd for rd in matrix_query.features]

@pytest.mark.parametrize('matrix', matrix_list)
def test_all_matrix(matrix: ConnectivityMatrix):
    basic_result = JSONEncoder.encode(matrix, nested=True, detail=False)
    ConnectivityMatrix.SiibraSerializationSchema(**basic_result)
    assert 'matrix' not in basic_result

    full_result = JSONEncoder.encode(matrix, nested=True, detail=True)
    ConnectivityMatrix.SiibraSerializationSchema(**full_result)
    assert 'matrix' in full_result
    assert full_result['matrix'].keys()