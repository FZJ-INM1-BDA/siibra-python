from typing import List
import pytest
import siibra
from siibra.core.json_encoder import JSONEncoder
from siibra.features.ieeg import IEEG_Session
from siibra.core import Atlas, Region
from ..util import get_model

queries = siibra.features.FeatureQuery.queries('ieeg')

# assert only one type of receptor query is returned
assert len(queries) == 1

ieeg_query = queries[0]
ieeg_list: List[IEEG_Session] = [rd for rd in ieeg_query.features]

@pytest.mark.parametrize('ieeg', ieeg_list)
def test_all_ieeg(ieeg: IEEG_Session):
    Model = get_model(IEEG_Session)

    basic_result = JSONEncoder.encode(ieeg, detail=False, nested=True)
    Model(**basic_result)
    assert 'detail' not in basic_result

    detail_result = JSONEncoder.encode(ieeg, detail=True, nested=True)
    Model(**detail_result)
    assert 'detail' in detail_result
    assert 'in_roi' not in detail_result.get('detail', {})

    human_atlas: Atlas = siibra.atlases['human']
    hoc1_right: Region = human_atlas.get_region('hoc1 right', parcellation='2 9')
    roi_result = JSONEncoder.encode(ieeg, region=hoc1_right, detail=True, nested=True)
    Model(**roi_result)
    assert 'in_roi' in roi_result.get('detail', {})
