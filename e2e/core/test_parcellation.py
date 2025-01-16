import siibra
import pytest
from siibra.core.parcellation import Parcellation

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


def test_parc_id_uniqueness():
    parcs = {p.name: p.id for p in siibra.parcellations}
    assert len(set(parcs.values())) == len(parcs)


XFAIL_LIST = [
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v1.01)', 'corticofugal pathways'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v1.01)', 'medial lemniscus'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v1.01)', 'facial nerve'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v1.01)', 'spinal cord'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v2)', 'corticofugal pathways'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v2)', 'medial lemniscus'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v2)', 'facial nerve'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v2)', 'spinal cord'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'anterior commissure'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'medial lemniscus'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'facial nerve'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'lateral lemniscus'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'neocortex'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'thalamus'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'brainstem'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'ventricular system'),
    ('Waxholm Space atlas of the Sprague Dawley rat brain (v3)', 'spinal cord'),
]


@pytest.mark.parametrize("parcellation", list(siibra.parcellations))
def test_region_id_uniqueness(parcellation: Parcellation):
    ids = set()
    duplicates = set()
    for region in parcellation:
        try:
            assert region.id not in ids, f"'{region.id}' has a duplicate in {parcellation.name}."
        except AssertionError:
            if (parcellation.name, region.name.lower()) in XFAIL_LIST:
                pytest.xfail(f"'{region.id}' in {parcellation.name} is expected to fail.")
            duplicates.add(region.id)
        ids.add(region)

    assert len(duplicates) == 0, f"Following regions a duplicate IDs in {parcellation.name}:\n{duplicates}"
