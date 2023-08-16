import siibra
import pytest

args = [
    ("waxholm v3", "neocortex", lambda region: region.name == "neocortex"),
    ("julich brain 1.18", "Ch 123 (Basal Forebrain) - left hemisphere", ValueError)
]

@pytest.mark.parametrize("parc_spec,reg_spec,validator", args)
def test_get_region(parc_spec,reg_spec,validator):
    if callable(validator):
        result = siibra.get_region(parc_spec, reg_spec)
        assert callable(validator)
        assert validator(result)
        return
    
    if issubclass(validator, Exception):
        with pytest.raises(validator):
            siibra.get_region(parc_spec, reg_spec)
        return
