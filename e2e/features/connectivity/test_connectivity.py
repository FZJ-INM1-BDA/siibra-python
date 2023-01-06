import siibra
import pytest
from typing import List

from siibra.features._basetypes.regional_connectivity import RegionalConnectivity

jba_29 = siibra.parcellations["2.9"]

args = [
    (jba_29, siibra.features.connectivity.FunctionalConnectivity),
    (jba_29, siibra.features.connectivity.StreamlineCounts),
    (jba_29, siibra.features.connectivity.StreamlineLengths),
    (jba_29, siibra.features.connectivity.ALL),
]

@pytest.mark.parametrize("concept,query_arg", args)
def test_connectivity(concept, query_arg):
    features: List['RegionalConnectivity'] = siibra.features.get(concept, query_arg)
    assert len(features) > 0

    for f in features:
        assert isinstance(f, RegionalConnectivity)
        assert len(f.subjects) > 0
        for subject in f.subjects:
            matrix = f.get_matrix(subject)
