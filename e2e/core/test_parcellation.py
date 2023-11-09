import siibra
import pytest

args = [
    ("waxholm v3", "neocortex", lambda region: region.name == "neocortex"),
    ("julich brain 1.18", "Ch 123 (Basal Forebrain) - left hemisphere", lambda region: region.name == "Ch 123 (Basal Forebrain) left"),
    ("julich brain 3", "v1", lambda region: region.name == "Area hOc1 (V1, 17, CalcS)"),
]


@pytest.mark.parametrize("parc_spec,reg_spec,validator", args)
def test_get_region(parc_spec, reg_spec, validator):

    subclasses_exception = isinstance(validator, type) and issubclass(validator, Exception)
    if callable(validator) and not subclasses_exception:
        result = siibra.get_region(parc_spec, reg_spec)
        assert callable(validator)
        assert validator(result)
        return

    if subclasses_exception:
        with pytest.raises(validator) as e:
            siibra.get_region(parc_spec, reg_spec)

    raise e("Should be either Exception or callable")
