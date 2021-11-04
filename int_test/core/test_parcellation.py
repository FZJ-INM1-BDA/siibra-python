from typing import List, Tuple
from siibra.core import Parcellation, Atlas
from siibra.core.json_encoder import JSONEncoder
import pytest
import siibra 
from pydantic import ValidationError, BaseModel

# should test all combinations of atlas/parellation
parameters: List[Tuple[str, str]] = [
    (atlas.id, parcellation.id)
    for atlas in siibra.atlases
    for parcellation in atlas.parcellations
]

def get_model():
    if issubclass(Parcellation.SiibraSerializationSchema, BaseModel):
        Model = Parcellation.SiibraSerializationSchema
    else:
        raise ValueError('SiibraSerializationSchema needs to extend pydantic.BaseModel')
    return Model

def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model()
        Model(id='test')

@pytest.mark.parametrize('atlas_id,parcellation_id', parameters)
def test_jsonable(atlas_id: str, parcellation_id: str):
    atlas: Atlas = siibra.atlases[atlas_id]
    parcellation: Parcellation = atlas.parcellations[parcellation_id]

    Model = get_model()
    result_json = JSONEncoder.encode(parcellation, nested=True)

    Model(**result_json)
    