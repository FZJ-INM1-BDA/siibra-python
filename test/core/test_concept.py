from siibra.core import concept
from unittest.mock import MagicMock
from siibra.retrieval.exceptions import NoSiibraConfigMirrorsAvailableException


import pytest


def test_provide_registry_tries_mirrors():
    
    assert len(concept._BOOTSTRAP_CONNECTORS) == 2, f"expect exactly 2 mirrors"

    for connector in concept._BOOTSTRAP_CONNECTORS:
        connector.get_loaders = MagicMock()
        connector.get_loaders.side_effect = Exception("mock request exception")
    
    class DummyCls:
        _bootstrap_folder = "dummy_folder"
        pass

    with pytest.raises(NoSiibraConfigMirrorsAvailableException):
        concept.provide_registry(DummyCls)

    for connector in concept._BOOTSTRAP_CONNECTORS:
        connector.get_loaders.assert_called()
