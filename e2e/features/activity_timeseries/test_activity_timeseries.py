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
    for f in features:
        assert isinstance(f, RegionalBOLD)
        assert len(f.table_keys) > 0
        assert all(isinstance(subject, str) for subject in f.table_keys)
        for subject in f.table_keys:
            f.get_table(subject)


args = [(jba_29, "RegionalBOLD"), (jba_29, RegionalBOLD)]


@pytest.mark.parametrize("concept,query_arg", args)
def test_get_connectivity(concept, query_arg):
    features: List["RegionalBOLD"] = siibra.features.get(concept, query_arg)
    assert len(features) > 0, f"Expecting some features exist, but none exist."
