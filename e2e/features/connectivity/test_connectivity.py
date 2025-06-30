import siibra
import pytest
from typing import List
from siibra.features.feature import CompoundFeature
from siibra.features.connectivity.regional_connectivity import RegionalConnectivity
from zipfile import ZipFile
import os
from time import time

from numpy import random

RANDOM_SEED = int(os.getenv("RANDOM_SEED", time()))
RANDOM_TEST_COUNT = int(os.getenv("RANDOM_TEST_COUNT", 10))
random.seed(int(RANDOM_SEED))

jba_29 = siibra.parcellations["julich 2.9"]
jba_303 = siibra.parcellations["julich 3.0.3"]

all_conn_instances = [
    f
    for Cls in siibra.features.feature.Feature._SUBCLASSES[RegionalConnectivity]
    for f in Cls._get_instances()
]


def test_copy_is_returned():
    feat: RegionalConnectivity = all_conn_instances[0]
    # retrieve matrix
    matrix = feat.data

    # ensure new val to be put is different from prev val
    prev_val = matrix.iloc[0, 0]
    new_val = 42
    assert new_val != prev_val
    matrix.iloc[0, 0] = new_val

    # retrieve matrix again
    matrix_again = feat.data
    assert matrix_again.iloc[0, 0] == prev_val


compound_conns = siibra.features.get(jba_303, RegionalConnectivity)


@pytest.mark.parametrize("cf", compound_conns)
def test_connectivity_get_data(cf: CompoundFeature):
    assert isinstance(cf, CompoundFeature)
    assert all([isinstance(f, RegionalConnectivity) for f in cf])
    assert len(cf.indices) > 0
    _ = cf.data
    subset = set(random.randint(0, len(cf), RANDOM_TEST_COUNT))
    for i in subset:
        f = cf[i]
        assert isinstance(f, RegionalConnectivity)
        matrix_idx_df = f.data
        assert all(
            matrix_idx_df.index[i] == r for i, r in enumerate(matrix_idx_df.columns)
        )


get_connectivity_args = [
    (jba_29, "StreamlineCounts"),
    (jba_29, "RegionalConnectivity"),
    (jba_303, "RegionalConnectivity"),
    (jba_303, "Anatomo"),
    (jba_29, RegionalConnectivity),
    (jba_29, "connectivity"),
    (jba_29, siibra.features.connectivity.StreamlineCounts),
    (jba_29, siibra.features.connectivity.StreamlineLengths),
]


@pytest.mark.parametrize("concept,query_arg", get_connectivity_args)
def test_get_connectivity(concept, query_arg):
    features: List["CompoundFeature"] = siibra.features.get(concept, query_arg)
    assert len(features) > 0, "Expecting some features exist, but none exist."
    assert all(issubclass(cf.feature_type, RegionalConnectivity) for cf in features)


def test_to_zip():
    # for now, only test the first feature, given the ci resource concern
    feat: RegionalConnectivity = all_conn_instances[0]
    feat.to_zip("file.zip")
    with ZipFile("file.zip") as z:
        filenames = [info.filename for info in z.filelist]
        assert len([f for f in filenames if f.endswith(".csv")]) == 1
    os.remove("file.zip")

    cf: CompoundFeature = compound_conns[0]
    cf.to_zip("file_compound.zip")
    with ZipFile("file_compound.zip") as cz:
        filenames = [info.filename for info in cz.filelist]
        assert len([f for f in filenames if f.endswith(".csv")]) == len(cf)
    os.remove("file_compound.zip")
