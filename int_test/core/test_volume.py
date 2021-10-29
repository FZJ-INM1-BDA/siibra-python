
from siibra.volumes import VolumeSrc
import pytest

from pydantic import ValidationError, BaseModel

from siibra.core.jsonable import JSONableConcept

# This should test possible parcellation.infos

def get_model(cls):
    assert issubclass(cls, JSONableConcept)
    if issubclass(cls.typed_json_output, BaseModel):
        Model = cls.typed_json_output
    else:
        raise ValueError('typed_json_output needs to extend pydantic.BaseModel')
    return Model

def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model(VolumeSrc)
        Model(id='test')

# the integration of jsonable in volume should already be tested in test_datasets.py