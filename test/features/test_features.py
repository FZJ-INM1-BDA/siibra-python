import pytest
import siibra
from unittest.mock import MagicMock, patch
from siibra.commons import MapType
from siibra.features.basetypes.feature import SpatialFeature
import random

def test_spatial_feature_statistical_map():
    
    class ExpectedException(Exception): pass

    r = siibra.parcellations['2.9'].get_region('fp1 left')

    with patch.object(r, 'fetch_regional_map', MagicMock()) as mock:
        mock.side_effect = ExpectedException

        space_of_interest = siibra.spaces['mni152']
        point = siibra.core.space.Point((0.0, 0.0, 0.0), space_of_interest)
        random_threshold = random.random()

        spatial_feature = SpatialFeature(point)
        try:
            spatial_feature.match(r, maptype=MapType.STATISTICAL, threshold_statistical=random_threshold)
            raise Exception(f"fetch_regional_map should have raised ExpectedException, but did not.")
        except ExpectedException:
            r.fetch_regional_map.assert_called_once_with(space=space_of_interest, maptype=MapType.STATISTICAL, threshold_statistical=random_threshold)
        except Exception as e:
            raise e

def test_gene_exp():
    r = siibra.parcellations['2.9'].get_region('fp1 left')
    args = (r, "gene")
    kwargs = {
        'gene': "MAOA",
        'maptype': MapType.STATISTICAL
    }
    features_higher = siibra.features.get(*args, threshold_statistical=0.9, **kwargs)
    features_lower = siibra.features.get(*args, threshold_statistical=0.2, **kwargs)

    # Using higher threshold == smaller number of features
    assert len(features_lower) > len(features_higher)
