from typing import List, Tuple
from siibra.core import Region, Atlas
from siibra.core.json_encoder import JSONEncoder
import pytest
import siibra
from siibra.core.parcellation import Parcellation
from pydantic import ValidationError, BaseModel

# ... maybe... not a good idea to test all possible regions in all possible parcellation in all possible atlas?
parameters: List[Tuple[str, str, str]]=[
    ('human', '2 9', 'hoc1 left'),
    ('monkey', 'primate', 'Left-Putamen'),
    ('rat', '4', 'external medullary lamina, auditory radiation'),
    ('mouse', '2017', 'Frontal pole, layer 2/3')
]

# it seems there are some ways in references are tracked in json_encoder
# seems to always stop at a certain point, raising circular reference
# parameters: List[Tuple[str, str, str]] = [
#     (atlas.id, parcellation.id, region.name)
#     for atlas in siibra.atlases
#     for parcellation in atlas.parcellations
#     for region in parcellation.regiontree
# ]

def get_model():
    if issubclass(Region.SiibraSerializationSchema, BaseModel):
        Model = Region.SiibraSerializationSchema
    else:
        raise ValueError('SiibraSerializationSchema needs to extend pydantic.BaseModel')
    return Model

def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model()
        Model(id='test')

@pytest.mark.parametrize('atlas_id,parcellation_id,region_id', parameters)
def test_region_jsonable(atlas_id: str, parcellation_id: str, region_id: str):
    atlas: Atlas = siibra.atlases[atlas_id]
    parcellation: Parcellation = atlas.parcellations[parcellation_id]
    region: Region = parcellation.decode_region(region_id)

    Model = get_model()
    result_json = JSONEncoder.encode(region, nested=True, depth_threshold=1000)
    Model(**result_json)

def test_region_has_children():
    atlas: Atlas = siibra.atlases['human']
    parcellation: Parcellation = atlas.parcellations['2 9']
    region: Region = parcellation.decode_region('hoc1')

    Model = get_model()

    result_json = JSONEncoder.encode(region, nested=True)
    region = Model(**result_json)
    assert len(region.children) == 2
    assert all([len(r.children) == 0 for r in region.children])

def test_region_filtered_by_space():

    atlas: Atlas = siibra.atlases['human']
    parcellation: Parcellation = atlas.parcellations['2 9']
    region: Region = parcellation.decode_region('hoc1')
    bigbrain = siibra.spaces['big brain']

    Model = get_model()

    result_json = JSONEncoder.encode(region, space=bigbrain, nested=True)
    region = Model(**result_json)
    assert len(region.children) == 0