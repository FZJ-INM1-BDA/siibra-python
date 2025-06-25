import pytest
from unittest.mock import patch, Mock
from siibra.commons import Species
from siibra.core.region import Region
from siibra.features.anchor import AnatomicalAnchor


@pytest.fixture
def fixture_teardown():
    yield ""
    AnatomicalAnchor._MATCH_MEMO = {}


region_specs = [
    ("foo", fixture_teardown),
    (Region("bar"), fixture_teardown),
]


@pytest.mark.parametrize("region_spec,teardown", region_specs)
@patch("siibra.features.anchor.find_regions")
def test_region_region_spec(mock_find_regions: Mock, region_spec, teardown):
    mock_find_regions.return_value = [Region("baz"), Region("hello world")]
    species = Species.UNSPECIFIED_SPECIES
    for r in mock_find_regions.return_value:
        r._species_cached = species

    anchor = AnatomicalAnchor(species, region=region_spec)
    assert isinstance(anchor.regions, dict)
    for _region in anchor.regions:
        assert isinstance(_region, Region)

    if isinstance(region_spec, Region):
        mock_find_regions.assert_not_called()
    elif isinstance(region_spec, str):
        mock_find_regions.assert_called_once_with(region_spec, filter_children=True, find_topmost=False)
    else:
        assert False, "Cannot have region as neither str or Region"
