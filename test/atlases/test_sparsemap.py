import siibra
import pytest
from unittest.mock import patch, PropertyMock, MagicMock
import re

@pytest.fixture
def mock_map_regions():
    yield ["foo", "get_region_return"]

@pytest.fixture
def mock_map_fetch():
    with patch.object(siibra.atlases.Map, "fetch") as mocked_fetch:
        yield mocked_fetch


foo_name = siibra.attributes.descriptions.Name(value="foo")
foo_id = siibra.attributes.descriptions.ID(value="foo")
foo_species = siibra.attributes.descriptions.SpeciesSpec(value="foo")


@pytest.fixture
def sparsemap(mock_map_fetch, mock_map_regions):
    smap = siibra.atlases.SparseMap(space_id="foo",
                                    maptype="labelled",
                                    attributes=(foo_name, foo_id, foo_species))
    parcellation_mock = MagicMock()
    with patch.object(siibra.atlases.SparseMap,
                      "regions",
                      new_callable=PropertyMock,
                      return_value=mock_map_regions):
        with patch.object(siibra.atlases.SparseMap,
                          "parcellation",
                          new_callable=PropertyMock,
                          return_value=parcellation_mock) as mock_mp_parcellation:
            yield (
                smap,
                mock_map_fetch,
                mock_mp_parcellation,
                parcellation_mock.get_region
            )

default_fetch_kwarg = {
    "frmt": None,
    "bbox": None,
    "resolution_mm": None,
    "color_channel": None,
    "max_download_GB": 0.2
}

region = siibra.Region(attributes=[foo_name, foo_id, foo_species], children=[])

fetch_args_kwargs = [
    ([], {}, RuntimeError, [], {}),
    ([12], {}, AssertionError, [], {}),
    ([region], {}, None, [], {"region": "foo", **default_fetch_kwarg}),
]


@pytest.mark.parametrize("args, kwargs, error, expected_args, default_fetch_kwarg", fetch_args_kwargs)
def test_sparsemap_fetch(args, kwargs, error, expected_args, default_fetch_kwarg, sparsemap):
    smap, mock, mpparc, parc_getregion = sparsemap
    if error:
        with pytest.raises(error):
            smap.fetch(*args, **kwargs)
        return
    
    mock.return_value = "foo"
    assert smap.fetch(*args, **kwargs) == "foo"
    
    kwargs = {**kwargs, **default_fetch_kwarg}
    mock.assert_called_once_with(*expected_args, **kwargs)

re_pattern = re.compile(r"foobar")

@pytest.mark.parametrize("args, kwargs, expected_get_region_arg", [
    ([region], {}, None),
    (["foo bar"], {}, "foo bar"),
    ([re_pattern], {}, re_pattern),
])
def test_sparsemap_fetch_w_str(args, kwargs, expected_get_region_arg, sparsemap, mock_map_regions):
    smap, mock, mpparc, parc_getregion = sparsemap
    
    mock.return_value = "foo"
    get_region_return = MagicMock()
    parc_getregion.return_value = get_region_return
    get_region_return.name = mock_map_regions[1]

    assert smap.fetch(*args, **kwargs) == "foo"
    if not expected_get_region_arg:
        mpparc.assert_not_called()
        parc_getregion.assert_not_called()
        return
    mpparc.assert_called_once()
    parc_getregion.assert_called_once_with(expected_get_region_arg)
    mock.asset_called_once_with({
        "region": get_region_return.name, 
        **default_fetch_kwarg
    })