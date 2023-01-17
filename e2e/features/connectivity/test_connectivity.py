import siibra
import pytest
from typing import List

from siibra.features.connectivity.regional_connectivity import RegionalConnectivity

jba_29 = siibra.parcellations["2.9"]

args = [
    pytest.param(
        jba_29, siibra.features.connectivity.FunctionalConnectivity,
        marks=pytest.mark.xfail(reason="Non-quadratic connectivity matrix 294x1")
    ),
    (jba_29, siibra.features.connectivity.StreamlineCounts),
    (jba_29, siibra.features.connectivity.StreamlineLengths),
    pytest.param(
        jba_29, siibra.features.connectivity.ALL,
        marks=pytest.mark.xfail(reasonn="Non-quadratic connectivity matrix 294x1")
    )
]


@pytest.mark.parametrize("concept,query_arg", args)
def test_connectivity(concept, query_arg):
    features: List['RegionalConnectivity'] = siibra.features.get(concept, query_arg)
    assert len(features) > 0

    for f in features:
        assert isinstance(f, RegionalConnectivity)
        assert len(f.subjects) > 0
        assert all(isinstance(subject, str) for subject in f.subjects)
        for subject in f.subjects:
            matrix = f.get_matrix(subject)
