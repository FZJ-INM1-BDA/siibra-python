import pytest
from unittest.mock import patch, PropertyMock

from siibra.commons.enum import ContainedInEnum
from siibra.exceptions import SiibraException
from siibra.atlases.parcellationmap import Map, VolumeFormats
from siibra.attributes.descriptions import ID, Name, SpeciesSpec


class MockCategory(ContainedInEnum):
    CAT1 = "cat1"
    CAT2 = "cat2"
    CAT3 = "cat3"


INST_FORMATS = ["foo1", "bar1"]
CATEGORY = MockCategory
LOOKUP = {
    None: ["foo0", "foo1", "bar0", "bar1", "bazz0", "bazz1"],
    MockCategory.CAT1.value: ["foo0", "foo1"],
    MockCategory.CAT2.value: ["bar0", "bar1"],
    MockCategory.CAT3.value: ["bazz0", "bazz1"],
}

args = [
    (None, "foo1", None),
    ("foo1", "foo1", None),
    ("bar1", "bar1", None),
    ("cat1", "foo1", None),
    ("cat2", "bar1", None),
    ("cat3", None, SiibraException),
    ("bar0", None, SiibraException),
    ("foo0", None, SiibraException),
    ("bazz0", None, SiibraException),
    ("bazz1", None, SiibraException),
]


@pytest.fixture
def mock_props_formats():
    with patch.object(Map, "formats", new_callable=PropertyMock) as formats_mock:
        formats_mock.return_value = INST_FORMATS
        yield formats_mock


@pytest.fixture
def mock_volume_formats():
    prev_cat = VolumeFormats.Category
    prev_lookup = VolumeFormats.FORMAT_LOOKUP
    VolumeFormats.Category = CATEGORY
    VolumeFormats.FORMAT_LOOKUP = LOOKUP
    yield
    VolumeFormats.Category = prev_cat
    VolumeFormats.FORMAT_LOOKUP = prev_lookup


@pytest.fixture
def map_inst():
    yield Map(
        space_id="space_id",
        parcellation_id="parc_id",
        maptype="labelled",
        region_mapping={"region": []},
        attributes=[ID(value="id"), Name(value="name"), SpeciesSpec(value="species")],
    )


@pytest.mark.parametrize("input_format, expected_format, Exc", args)
def test_select_format(
    input_format,
    expected_format,
    Exc,
    mock_props_formats,
    mock_volume_formats,
    map_inst,
):
    if Exc is not None:
        with pytest.raises(Exc):
            map_inst._select_format(input_format)
        return
    assert map_inst._select_format(input_format) == expected_format
