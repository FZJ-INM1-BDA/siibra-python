from typing import List, Tuple, Union
from siibra.core import Space, Atlas
from siibra.core.json_encoder import JSONEncoder
import pytest
import siibra
from pydantic import ValidationError, BaseModel

from siibra.core.space import Point

# This should always go through all available spaces in all atlases
parameters: List[Tuple[str, str]] = [
    (atlas.id, space.id)
    for atlas in siibra.atlases
    for space in atlas.spaces
]
def get_model():
    if issubclass(Space.typed_json_output, BaseModel):
        Model = Space.typed_json_output
    else:
        raise ValueError('typed_json_output needs to extend pydantic.BaseModel')
    return Model
def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model()
        Model(id='test')
@pytest.mark.parametrize('atlas_id,space_id', parameters)
def test_region_jsonable(atlas_id: str, space_id: str):
    atlas: Atlas = siibra.atlases[atlas_id]
    space: Space = atlas.spaces[space_id]

    Model = get_model()
    result_json = JSONEncoder.encode(space, nested=True)
    Model(**result_json)


# Point
space = siibra.spaces['mni152']
point_specs: List[Union[str, tuple, list]] = [
    '-12.950mm, 29.750mm, -4.200mm'
]
def get_point_model():
    if issubclass(Point.typed_json_output, BaseModel):
        Model = Point.typed_json_output
    else:
        raise ValueError('typed_json_output needs to extend pydantic.BaseModel')
    return Model
    
@pytest.mark.parametrize('point_spec', point_specs)
def test_point_jsonable(point_spec):
    p = Point(Point.parse(point_spec), space)
    Model = get_point_model()
    result_json = JSONEncoder.encode(p, nested=True)

    Model(**result_json)