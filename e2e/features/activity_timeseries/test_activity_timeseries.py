import siibra
import pytest
from typing import List

from siibra.features.tabular.regional_timeseries_activity import RegionalBOLD
from e2e.util import check_duplicate

jba_29 = siibra.parcellations["julich 2.9"]


def test_id_unique():
    features = siibra.features.get(jba_29, RegionalBOLD)
    duplicates = check_duplicate([f.id for f in features])
    assert len(duplicates) == 0


def test_feature_unique():
    features = siibra.features.get(jba_29, RegionalBOLD)
    duplicates = check_duplicate([f for f in features])
    assert len(duplicates) == 0


# the get_table is a rather expensive operation
# only do once for the master list
def test_timeseries_get_table():
    features: List["RegionalBOLD"] = siibra.features.get(jba_29, "bold")
    assert len(features) > 0
    for cf in features:
        assert cf.subfeature_type == RegionalBOLD
        assert len(cf) > 0
        assert all(isinstance(fi.table_keys, str) for fi in cf)
        _ = cf.data
        for fi in cf:
            _ = fi.data


args = [(jba_29, "RegionalBOLD"), (jba_29, RegionalBOLD)]


@pytest.mark.parametrize("concept,query_arg", args)
def test_get_bold(concept, query_arg):
    features: List["RegionalBOLD"] = siibra.features.get(concept, query_arg)
    assert len(features) > 0, "Expecting some features exist, but none exist."
