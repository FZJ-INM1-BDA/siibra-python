from typing import List, Tuple
from siibra.core import Atlas
from siibra.core.json_encoder import JSONEncoder
import pytest
import siibra
from pydantic import ValidationError, BaseModel

# should test all available atlases
parameters: List[str]=[
    atlas.id for atlas in siibra.atlases
]

def get_model():
    if issubclass(Atlas.typed_json_output, BaseModel):
        Model = Atlas.typed_json_output
    else:
        raise ValueError('typed_json_output needs to extend pydantic.BaseModel')
    return Model

def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model()
        Model(id='test')

@pytest.mark.parametrize('atlas_id', parameters)
def test_atlas_jsonable(atlas_id: str):
    atlas: Atlas = siibra.atlases[atlas_id]

    Model = get_model()
    result_json = JSONEncoder.encode(atlas, nested=True, depth_threshold = 1000)
    Model(**result_json)
    