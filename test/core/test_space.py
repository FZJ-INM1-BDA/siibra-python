import pytest
from siibra.core import Space, Point
from siibra.volumes.volume import File

all_spaces = [s for s in Space.REGISTRY]
def test_num_space():
    from siibra.core import Space, Point
    all_spaces = [s for s in Space.REGISTRY]
    assert len(all_spaces) > 5

@pytest.mark.parametrize('space', all_spaces)
def test_space_jsonable(space: Space):
    space.json()

@pytest.mark.parametrize('space', all_spaces)
def test_space_volumes_defined(space: Space):
    assert len(space.volumes) > 0
    assert all([
        isinstance(v, File) for v in  space.volumes
    ])

@pytest.mark.parametrize('space', all_spaces)
def test_space_volumes_jsonable(space: Space):
    for v in space.volumes:
        v.json()

point_json_1={
    "@id": 'test-id',
    "@type": "https://openminds.ebrains.eu/sands/CoordinatePoint",
    "https://openminds.ebrains.eu/vocab/coordinateSpace": {
        "@id": Space.parse_legacy_id("minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588")
    },
    "https://openminds.ebrains.eu/vocab/coordinates": [{
        "value": 10,
    }, {
        "value": 11,
    }, {
        "value": 12,
    }]
}

point_json_2={
    "@id": 'test-id-2',
    "@type": "https://openminds.ebrains.eu/sands/CoordinatePoint",
    "https://openminds.ebrains.eu/vocab/coordinateSpace": {
        "@id": Space.parse_legacy_id("minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992")
    },
    "https://openminds.ebrains.eu/vocab/coordinates": [{
        "value": 10,
    }, {
        "value": 11,
    }, {
        "value": 12,
    }]
}

def test_point_creation():
    p = Point(**point_json_1)
    assert p.coordinates_tuple == (10, 11, 12)
    assert p.id == point_json_1.get("@id")

subtraction = ['sub,expected,expect_raise', [
    (1, (9, 10, 11), False),
    (Point(
        [1,2,3],
        **point_json_1), (9,9,9), False),
    (Point(**point_json_2), None, True)
]]

@pytest.mark.parametrize(*subtraction)
def test_point_sub_num(sub,expected,expect_raise):
    p = Point(**point_json_1)
    if expect_raise:
        with pytest.raises(AssertionError):
            sub_result = p - sub
    else:
        sub_result = p - sub
        assert sub_result.coordinates_tuple == expected
        assert p.coordinates_tuple == (10, 11, 12)
        assert sub_result is not p


addition = ['add,expected,expect_raise', [
    (1, (11, 12, 13), False),
    (Point(
        [1,2,3],
        **point_json_1), (11, 13, 15), False),
    (Point(**point_json_2), None, True)
]]
@pytest.mark.parametrize(*addition)
def test_point_add_num(add,expected,expect_raise):
    p = Point(**point_json_1)
    if expect_raise:
        with pytest.raises(AssertionError):
            add_result = p + add
    else:
        add_result = p + add
        assert add_result.coordinates_tuple == expected
        assert p.coordinates_tuple == (10, 11, 12)
        assert add_result is not p

def test_point_div():
    p = Point(**point_json_1)
    new_p = p / 2
    assert new_p.coordinates_tuple == (5, 5.5, 6)
    assert p.coordinates_tuple == (10, 11, 12)
    assert new_p is not p


def test_point_mul():
    p = Point(**point_json_1)
    new_p = p * 2
    new_p_r = 2 * p
    assert new_p.coordinates_tuple == (20, 22, 24)
    assert new_p.coordinates_tuple == new_p_r.coordinates_tuple
    assert p.coordinates_tuple == (10, 11, 12)
    assert new_p is not p