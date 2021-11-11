from typing import List
from siibra.core import Atlas
from siibra.core.json_encoder import JSONEncoder
import pytest
import siibra
from ..util import get_model

# should test all available atlases
parameters: List[str]=[
    atlas.id for atlas in siibra.atlases
]

@pytest.mark.parametrize('atlas_id', parameters)
def test_atlas_jsonable(atlas_id: str):
    atlas: Atlas = siibra.atlases[atlas_id]

    Model = get_model(Atlas)
    result_json = JSONEncoder.encode(atlas, nested=True, depth_threshold = 1000)
    Model(**result_json)
    