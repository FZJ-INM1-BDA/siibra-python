import siibra
import pytest
from typing import List
import sys
from siibra.features.tabular.regional_timeseries_activity import RegionalBOLD
from siibra.features.feature import CompoundFeature

skip_on_windows = pytest.mark.skipif(sys.platform == "win32", reason="Fails due to memory limitation issues on Windows on Github actions. (Passes on local machines.)")

jba_29 = siibra.parcellations["julich 2.9"]

bold_cfs = [
    *siibra.features.get(jba_29, "bold"),
    *siibra.features.get(siibra.parcellations["julich 3"], "bold")
]


# getting data is a rather expensive operation
# only do once for the master list
@pytest.mark.parametrize("cf", bold_cfs)
@skip_on_windows
def test_timeseries_get_data(cf):
    assert isinstance(cf, CompoundFeature)
    for f in cf:
        assert isinstance(f, RegionalBOLD)
        assert isinstance(f.subject, str)
        _ = f.data


args = [(jba_29, "RegionalBOLD"), (jba_29, RegionalBOLD)]


@pytest.mark.parametrize("concept,query_arg", args)
def test_get_connectivity(concept, query_arg):
    features: List["CompoundFeature"] = siibra.features.get(concept, query_arg)
    assert len(features) > 0, "Expecting some features exist, but none exist."
