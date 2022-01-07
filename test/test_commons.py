from siibra.commons import TypedRegistry
from unittest.mock import Mock, call, patch
import pytest

def test_typed_registry_can_add():
    registry = TypedRegistry()
    assert len(registry) == 0
    registry.add('key', 1)
    assert len(registry) == 1

def test_typed_reg_contains():
    registry = TypedRegistry()
    registry.add('key', 1)
    assert 'key' in registry
    assert 1 in registry

def test_typed_reg_iter():
    registry = TypedRegistry()
    registry.add('key', 1)
    registry.add('key2', 2)
    assert [v for v in registry] == [1, 2]


def test_typed_reg_dir():
    mock_get_aliases = Mock()
    mock_get_aliases.side_effect = [
        ["alias_1"],
        ["alias_2"],
    ]
    registry = TypedRegistry(get_aliases=mock_get_aliases)
    registry.add('key', 1)
    registry.add('key2', 2)

    with patch.object(TypedRegistry, 'alias_to_attr') as patched_fn:
        
        patched_fn.side_effect = ["final_alias_1", "final_alias_2"]
        assert registry.__dir__() == ["final_alias_1", "final_alias_2"]
        
        mock_get_aliases.assert_has_calls([
            call("key", 1),
            call("key2", 2),
        ], any_order=False)

        patched_fn.assert_has_calls([
            call("alias_1"),
            call("alias_2"),
        ], any_order=False)


alias_to_attr_param = [
    ("hello world", "HELLO_WORLD"),
    ("test (1-1)", "TEST_1_1"),
]

@pytest.mark.parametrize("input,expected", alias_to_attr_param)
def test_typed_reg_alias_to_alias(input,expected):
    assert TypedRegistry.alias_to_attr(input) == expected

def test_typed_reg_find_str_key():
    registry = TypedRegistry(
        get_aliases=lambda key, value: [value.get("name")]
    )
    obj1 = { "name": "value1" }
    obj2 = { "name": "value2" }
    registry.add("key1", obj1)
    registry.add("key2", obj2)
    assert registry.find("key1") == [obj1]
    assert registry.find("key2") == [obj2]


def test_typed_reg_find_str_alias():
    registry = TypedRegistry(
        get_aliases=lambda key, value: [value.get("name")]
    )
    obj1 = { "name": "value1" }
    obj2 = { "name": "value 2" }
    registry.add("key1", obj1)
    registry.add("key2", obj2)
    assert registry.find("value1") == [obj1]
    assert registry.find("value2") == [obj2]

def test_typed_reg_find_dict():
    registry = TypedRegistry()
    obj1 = { "name": "name1" }
    obj2 = { "name": "name2" }
    registry.add("key1", obj1)
    registry.add("key2", obj2)

    assert registry.find({
        "@id": "key1"
    }) == [obj1]

    assert registry.find(obj1) == [obj1]

def test_typed_reg_find_int():

    registry = TypedRegistry()
    obj1 = { "name": "name1" }
    obj2 = { "name": "name2" }
    registry.add("key1", obj1)
    registry.add("key2", obj2)
    assert registry.find(0) == [obj1]
