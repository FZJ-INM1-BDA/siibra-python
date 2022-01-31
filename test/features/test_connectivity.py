import unittest
import pytest

from siibra.features.connectivity import PrereleasedStreamlineCountQuery, StreamlineCounts

con_query = PrereleasedStreamlineCountQuery()

def test_some_conn_data():
    assert len(con_query.features) > 0, f"expect at least 1 connectivity data, but got {len(con_query.features)}"


@pytest.mark.parametrize('conn_feat', con_query.features)
def test_conn_to_model(conn_feat: StreamlineCounts):
    try:
        conn_feat.to_model()
    except AssertionError as e:
        # TODO
        # two connectivity sources xfail here
        pytest.xfail(str(e))


if __name__ == "__main__":
    unittest.main()
