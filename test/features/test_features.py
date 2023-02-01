import pytest
import siibra
from unittest.mock import MagicMock, patch
from siibra.commons import MapType
import random


def test_gene_exp():
    r = siibra.parcellations['2.9'].get_region('fp1 left')
    args = (r, "gene")
    kwargs = {
        'gene': "MAOA",
        'maptype': MapType.STATISTICAL
    }
    features_higher = siibra.features.get(*args, threshold_statistical=0.9, **kwargs)
    features_lower = siibra.features.get(*args, threshold_statistical=0.2, **kwargs)

    # should have received one gene expression feature each
    assert len(features_higher) == 1
    assert len(features_lower) == 1

    # Using higher threshold should result in less gene expresssion measures
    assert len(features_lower[0].data) > len(features_higher[0].data)
