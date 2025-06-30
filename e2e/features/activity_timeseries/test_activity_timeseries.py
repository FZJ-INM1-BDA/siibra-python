import siibra
import pytest
from typing import List
import sys
import os
from siibra.features.tabular.regional_timeseries_activity import RegionalBOLD
from siibra.features.feature import CompoundFeature
from time import time

from numpy import random

RANDOM_SEED = int(os.getenv("RANDOM_SEED", time()))
RANDOM_TEST_COUNT = int(os.getenv("RANDOM_TEST_COUNT", 10))
random.seed(int(RANDOM_SEED))

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Fails due to memory limitation issues on Windows on Github actions. (Passes on local machines.)",
)

jba_29 = siibra.parcellations["julich 2.9"]
get_bold_args = [(jba_29, "RegionalBOLD"), (jba_29, RegionalBOLD)]


@pytest.mark.parametrize("concept,query_arg", get_bold_args)
def test_get_bold(concept, query_arg):
    features: List["CompoundFeature"] = siibra.features.get(concept, query_arg)
    assert len(features) > 0, "Expecting some features exist, but none exist."


bold_cfs = [
    *siibra.features.get(jba_29, "bold"),
    *siibra.features.get(siibra.parcellations["julich 3"], "bold"),
]


# getting data is a rather expensive operation
# only do once for the master list
@pytest.mark.parametrize("cf", bold_cfs)
@skip_on_windows
def test_timeseries_get_data(cf):
    assert isinstance(cf, CompoundFeature)
    subset = set(random.randint(0, len(cf), RANDOM_TEST_COUNT))
    _ = cf.data
    for i in subset:
        f = cf[i]
        assert isinstance(f, RegionalBOLD)
        assert isinstance(f.subject, str)
        _ = f.data
