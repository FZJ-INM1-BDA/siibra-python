from typing import List

import pytest
import siibra
from siibra.core.json_encoder import JSONEncoder
from siibra.features.voi import VolumeOfInterest
from ..util import get_model

queries = siibra.features.FeatureQuery.queries('VolumeOfInterest')

# assert only one type of receptor query is returned
assert len(queries) == 1

voi_query_obj = queries[0]
voi: List[VolumeOfInterest] = [rd for rd in voi_query_obj.features]

@pytest.mark.parametrize('voi', voi)
def test_all_vois(voi: VolumeOfInterest):
    Model = get_model(VolumeOfInterest)
    output = JSONEncoder.encode(voi, nested = True)
    Model(**output)
