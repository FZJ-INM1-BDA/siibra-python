from typing import List, Tuple
from siibra.core import Parcellation, Atlas
from siibra.core.json_encoder import JSONEncoder
import pytest
import siibra 
from pydantic import ValidationError
from int_test.util import get_model

# should test all combinations of atlas/parellation
parameters: List[Tuple[str, str]] = [
    (atlas.id, parcellation.id)
    for atlas in siibra.atlases
    for parcellation in atlas.parcellations
]


def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model(Parcellation)
        Model(id='test')

@pytest.mark.parametrize('atlas_id,parcellation_id', parameters)
def test_jsonable(atlas_id: str, parcellation_id: str):
    atlas: Atlas = siibra.atlases[atlas_id]
    parcellation: Parcellation = atlas.parcellations[parcellation_id]

    Model = get_model(Parcellation)
    result_json = JSONEncoder.encode(parcellation, nested=True)

    Model(**result_json)
    