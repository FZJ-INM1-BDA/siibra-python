import siibra
import pytest
from typing import List
import sys
from siibra.features.feature import CompoundFeature
from siibra.features.connectivity.regional_connectivity import RegionalConnectivity
from e2e.util import check_duplicate
from zipfile import ZipFile

pytestmark = pytest.mark.skipif(sys.platform == "ubuntu", reason="Fails due to memory limitation issues on Windows on Github actions. (Passes on local machines.)")

all_conn_instances = [
    f
    for Cls in siibra.features.feature.Feature.SUBCLASSES[RegionalConnectivity]
    for f in Cls.get_instances()
]

compound_conns = siibra.features.get(siibra.parcellations['julich 3'], RegionalConnectivity)


def test_id_unique():
    duplicates = check_duplicate([f.id for f in all_conn_instances])
    assert len(duplicates) == 0


def test_feature_unique():
    duplicates = check_duplicate([f for f in all_conn_instances])
    assert len(duplicates) == 0


@pytest.mark.parametrize("cf", compound_conns)
def test_connectivity_get_data(cf: CompoundFeature):
    assert isinstance(cf, CompoundFeature)
    assert all([isinstance(f, RegionalConnectivity) for f in cf])
    assert len(cf.indices) > 0
    matrix_df = cf.data  # get average
    assert all(matrix_df.index[i] == r for i, r in enumerate(matrix_df.columns))
    for f in cf:
        assert isinstance(f, RegionalConnectivity)
        matrix_idx_df = f.data
        assert all(matrix_idx_df.index[i] == r for i, r in enumerate(matrix_idx_df.columns))


jba_29 = siibra.parcellations["julich 2.9"]
jba_3 = siibra.parcellations["julich 3"]

args = [
    (jba_29, "StreamlineCounts"),
    (jba_29, "RegionalConnectivity"),
    (jba_3, "RegionalConnectivity"),
    (jba_3, "Anatomo"),
    (jba_29, RegionalConnectivity),
    (jba_29, "connectivity"),
    (jba_29, siibra.features.connectivity.StreamlineCounts),
    (jba_29, siibra.features.connectivity.StreamlineLengths),
]


@pytest.mark.parametrize("concept,query_arg", args)
def test_get_connectivity(concept, query_arg):
    features: List["CompoundFeature"] = siibra.features.get(concept, query_arg)
    assert len(features) > 0, "Expecting some features exist, but none exist."
    assert all(issubclass(cf.subfeature_type, RegionalConnectivity) for cf in features)


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


def test_export():
    # for now, only test the first feature, given the ci resource concern
    feat: RegionalConnectivity = all_conn_instances[0]
    feat.export("file.zip")
    z = ZipFile("file.zip")
    filenames = [info.filename for info in z.filelist]
    assert len([f for f in filenames if f.endswith(".csv")]) > 10
