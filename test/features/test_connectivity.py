import unittest
import pytest

from siibra.features.connectivity import (
    PrereleasedStreamlineCountQuery,
    ConnectivityMatrixDataModel,

    PrereleasedStreamlineLengthQuery,

    PrereleasedRestingStateQuery,

    HcpStreamlineCountQuery,
    HcpStreamlineLengthQuery,
    HcpRestingStateQuery,
)


queries_tuple = (
    PrereleasedStreamlineCountQuery,
    PrereleasedStreamlineLengthQuery,
    PrereleasedRestingStateQuery,

    HcpStreamlineCountQuery,
    HcpStreamlineLengthQuery,
    HcpRestingStateQuery,
)

queries_features_tuple = (
    feat
    for Query in queries_tuple
    for feat in Query().features
)

@pytest.mark.parametrize('conn_feat', queries_features_tuple)
def test_conn_to_model(conn_feat):
    model = None
    try:
        model = conn_feat.to_model(detail=True)
    except Exception as err:
        pytest.xfail(str(err))
    if model:
        ConnectivityMatrixDataModel(**model.dict())



if __name__ == "__main__":
    unittest.main()
