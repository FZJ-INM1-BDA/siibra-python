
from siibra.volumes import VolumeSrc
import pytest

from pydantic import ValidationError, BaseModel

from siibra.core.jsonable import SiibraSerializable
from ..util import get_model
# This should test possible parcellation.infos

def test_wrong_model_raises():
    with pytest.raises(ValidationError):
        Model = get_model(VolumeSrc)
        Model(id='test')

# the integration of jsonable in volume should already be tested in test_datasets.py