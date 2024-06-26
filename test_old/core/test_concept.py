from unittest.mock import MagicMock, PropertyMock
from unittest.mock import patch
from uuid import uuid4
import pytest

from siibra.configuration.configuration import Configuration
from siibra.core.concept import AtlasConcept, InstanceTable, _REGISTRIES
from siibra.commons import Species

CONF_FOLDER = "CONF_FOLDER"
IDENTIFIER = "my-id"
NAME = "my-name"


class DummyItem:
    def __init__(self):
        self.key = str(uuid4())

    @classmethod
    def match(cls):
        return False


@pytest.fixture
def DummyClsKwarg():

    class DummyClsKwarg(AtlasConcept, configuration_folder=CONF_FOLDER):
        pass

    _REGISTRIES[DummyClsKwarg] = None
    yield DummyClsKwarg

    _REGISTRIES.pop(DummyClsKwarg, None)


def test_init(DummyClsKwarg):

    assert issubclass(DummyClsKwarg, AtlasConcept)
    assert DummyClsKwarg._configuration_folder == CONF_FOLDER
    instance = DummyClsKwarg(
        identifier=IDENTIFIER, name=NAME, species=Species.HOMO_SAPIENS
    )
    assert isinstance(instance, AtlasConcept)


def test_class_registry_init(DummyClsKwarg):
    with patch.object(Configuration, "register_cleanup") as mock_register_cleanup:
        with patch.object(Configuration, "build_objects") as mock_build_objects:
            with patch.object(
                Configuration,
                "folders",
                new_callable=PropertyMock,
                return_value=[CONF_FOLDER],
            ) as mock_folder_prop:
                mock_build_objects.return_value = [DummyItem(), DummyItem()]
                reg = DummyClsKwarg.registry()
                assert isinstance(reg, InstanceTable)
                mock_folder_prop.assert_called_once()
                mock_build_objects.assert_called_once_with(CONF_FOLDER)
                mock_register_cleanup.assert_called_once_with(
                    DummyClsKwarg.clear_registry
                )


def test_class_registry_cached(DummyClsKwarg):
    dummy = DummyItem()
    table = InstanceTable(
        elements={dummy.key: dummy}, matchfunc=dummy.match
    )
    _REGISTRIES[DummyClsKwarg] = table
    assert DummyClsKwarg.registry() is table


def test_clear_registry(DummyClsKwarg):
    dummy = DummyItem()
    _REGISTRIES[DummyClsKwarg] = InstanceTable(
        elements={dummy.key: dummy}, matchfunc=dummy.match
    )
    DummyClsKwarg.clear_registry()
    assert _REGISTRIES[DummyClsKwarg] is None


def test_get_instance(DummyClsKwarg):
    with patch.object(AtlasConcept, "registry") as registry_mock:
        spec = "foo-bar"
        dummy = DummyItem()
        dummy2 = DummyItem()
        registry_mock.return_value = dummy
        dummy.get = MagicMock(return_value=dummy2)
        get_instance_return = DummyClsKwarg.get_instance(spec)

        assert get_instance_return is dummy2
        registry_mock.assert_called()
        dummy.get.assert_called_once_with(spec)
