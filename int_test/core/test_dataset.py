from typing import List, Tuple
from siibra.core import Parcellation, Atlas
from siibra.core.datasets import Dataset
from siibra.core.json_encoder import JSONEncoder
import pytest
import siibra 
from pydantic import ValidationError, BaseModel

from int_test.util import get_model

# This should test possible parcellation.infos
parameters: List[Tuple[str, str, str]]=[
    (atlas.id, parc.id)
    for atlas in siibra.atlases
    for parc in atlas.parcellations]

def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model(Dataset)
        Model(id='test')

@pytest.mark.parametrize('atlas_id,parcellation_id', parameters)
def test_jsonable(atlas_id: str, parcellation_id: str):
    atlas: Atlas = siibra.atlases[atlas_id]
    parcellation: Parcellation = atlas.parcellations[parcellation_id]

    for info in parcellation.infos:
        Model = get_model(type(info))
    
        result_json = JSONEncoder.encode(info, nested=True)
        try:
            Model(**result_json)
        except ValidationError as e:
            print(f'error validating "{str(parcellation)}" info: with type: {type(info)} {str(info)}')
            raise e
    