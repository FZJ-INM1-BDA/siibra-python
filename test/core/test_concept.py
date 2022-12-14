from siibra.core import concept
from unittest.mock import MagicMock, PropertyMock, call
from siibra.retrieval.exceptions import NoSiibraConfigMirrorsAvailableException
from urllib3.exceptions import NewConnectionError
from siibra.core.concept import AtlasConcept, InstanceTable
import siibra
from unittest.mock import patch
from uuid import uuid4
import unittest

CONF_FOLDER = "CONF_FOLDER"
IDENTIFIER = "my-id"
NAME = "my-name"

class DummyClsKwarg(AtlasConcept, configuration_folder=CONF_FOLDER):
    pass

class DummyItem:
    def __init__(self):
        self.key = str(uuid4())
    @classmethod
    def match(cls):
        return False

class TestAtlasConcept(unittest.TestCase):

    def setUp(self):
        DummyClsKwarg._registry_cached = None

    def test_init(self):
        assert issubclass(DummyClsKwarg, AtlasConcept)
        assert DummyClsKwarg._configuration_folder == CONF_FOLDER
        instance = DummyClsKwarg(identifier=IDENTIFIER, name=NAME)
        assert isinstance(instance, AtlasConcept)

    def test_class_registry_init(self):
        with patch.object(siibra.configuration.Configuration, 'register_cleanup') as mock_register_cleanup:
            with patch.object(siibra.configuration.Configuration, 'build_objects') as mock_build_objects:
                with patch.object(siibra.configuration.Configuration, 'folders', new_callable=PropertyMock, return_value=[CONF_FOLDER]) as mock_folder_prop:
                    mock_build_objects.return_value = [DummyItem(), DummyItem()]
                    reg = DummyClsKwarg.registry()
                    assert isinstance(reg, InstanceTable)
                    mock_folder_prop.assert_called_once()
                    mock_build_objects.assert_called_once_with(CONF_FOLDER)
                    mock_register_cleanup.assert_called_once_with(DummyClsKwarg.clear_registry)

    def test_class_registry_cached(self):
        dummy = DummyItem()
        DummyClsKwarg._registry_cached = dummy
        assert DummyClsKwarg.registry() is dummy
    
    def test_clear_registry(self):
        dummy = DummyItem()
        DummyClsKwarg._registry_cached = dummy
        DummyClsKwarg.clear_registry()
        assert DummyClsKwarg._registry_cached is None
    
    def test_get_instance(self):
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
