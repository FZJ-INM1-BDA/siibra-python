import pytest

import siibra

JBA31_ID = "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-310"
JBA30_ID = "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-300"
JBA29_ID = "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290"
JBA118_ID = "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579"

CH123_LH = "Ch 123 (Basal Forebrain) - left hemisphere"
HOC1 = "Area hOc1 (V1, 17, CalcS)"
WAXHOLM_V3_ID = "minds/core/parcellationatlas/v1.0.0/ebb923ba-b4d5-4b82-8088-fa9215c2e1fe"
WAXHOLM_V4_ID = "minds/core/parcellationatlas/v1.0.0/ebb923ba-b4d5-4b82-8088-fa9215c2e1fe-v4"

@pytest.mark.parametrize(
    "parcellation_spec, regionspec, expected_parc_id, expected_region_name, error",
    [
        ("julich", "hoc1", JBA31_ID, HOC1, None),
        ("2.9", "hoc1", JBA29_ID, HOC1, None),
        ("waxholm v3", "neocortex", WAXHOLM_V3_ID, "neocortex", None),
        ("julich brain 1.18", CH123_LH, JBA118_ID, CH123_LH, None),
        ("julich brain 3", "v1", None, None, siibra.exceptions.NotFoundException),
        ("julich brain 3.0", "v1", None, None, siibra.exceptions.NotFoundException),
        ("julich brain 3.0.3", "v1", JBA30_ID, HOC1, None),
        ("julich 3.0", "frontal lobe", None, None, siibra.exceptions.NotFoundException), 
        ("julich 3.0.3", "frontal lobe", JBA30_ID, "frontal lobe", None), 
        ("waxholm 4", "lateral olfactory tract", WAXHOLM_V4_ID, "lateral olfactory tract", None),
    ]
)
def test_get_region(parcellation_spec, regionspec, expected_parc_id, expected_region_name, error):
    if error:
        with pytest.raises(error):
            siibra.get_region(parcellation_spec, regionspec)
        return
    region = siibra.get_region(parcellation_spec, regionspec)
    assert region.parcellation.ID == expected_parc_id
    assert region.name == expected_region_name

@pytest.mark.parametrize(
    "search_str, expected_name, error",
    [
        ("2.9", "Julich-Brain Cytoarchitectonic Atlas (v2.9)", None),
        ("julich", "Julich-Brain Cytoarchitectonic Atlas (v3.1)", None),
        ("foobar", None, AssertionError),
    ]
)
def test_get_parcellation(search_str, expected_name, error):
    if error:
        with pytest.raises(error):
            siibra.get_parcellation(search_str)
        return
    result = siibra.get_parcellation(search_str)
    assert result.name == expected_name
