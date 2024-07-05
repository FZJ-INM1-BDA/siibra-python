import pytest
import siibra

JBA31_ID = "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-310"
JBA29_ID = "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290"
HOC1 = "Area hOc1 (V1, 17, CalcS)"

@pytest.mark.parametrize(
    "parcellation_spec, regionspec, expected_parc_id, expected_region_name, error",
    [
        ("julich", "hoc1", JBA31_ID, HOC1, None),
        ("2.9", "hoc1", JBA29_ID, HOC1, None),
    ]
)
def test_get_region(parcellation_spec, regionspec, expected_parc_id, expected_region_name, error):
    region = siibra.get_region(parcellation_spec, regionspec)
    assert region.parcellation.ID == expected_parc_id
    assert region.name == expected_region_name
