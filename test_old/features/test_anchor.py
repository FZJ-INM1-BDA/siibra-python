import pytest
from siibra.core.region import Region
from siibra.features.anchor import AnatomicalAnchor, Parcellation, Species
from unittest.mock import patch


@pytest.fixture
def fixture_teardown():
    yield ""
    AnatomicalAnchor._MATCH_MEMO = {}


region_specs = [
    ("foo", fixture_teardown),
    (Region("bar"), fixture_teardown),
]


@pytest.mark.parametrize("region,teardown", region_specs)
def test_region_region_spec(region, teardown):
    mock_found_regions = [Region("baz"), Region("hello world")]
    species = Species.UNSPECIFIED_SPECIES
    with patch.object(
        Parcellation, "find_regions", return_value=mock_found_regions
    ) as mock_find_regions:
        anchor = AnatomicalAnchor(species, region=region)
        assert isinstance(anchor.regions, dict)
        for _region in anchor.regions:
            assert isinstance(_region, Region)

        if isinstance(region, Region):
            mock_find_regions.assert_not_called()
        elif isinstance(region, str):
            mock_find_regions.assert_called_once_with(region, species)
        else:
            assert False, "Cannot have region as neither str or Region"
