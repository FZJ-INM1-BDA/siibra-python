from siibra.core import concept
from unittest.mock import MagicMock
from siibra.retrieval.exceptions import NoSiibraConfigMirrorsAvailableException
from urllib3.exceptions import NewConnectionError

import pytest


class DummyCls:
    _bootstrap_folder = "dummy_folder"
    pass


def test_number_of_connectors():
    assert len(concept._BOOTSTRAP_CONNECTORS) == 2, f"expect exactly 2 mirrors"


def test_provide_registry_tries_mirrors():
    for connector in concept._BOOTSTRAP_CONNECTORS:
        connector.get_loaders = MagicMock()
        connector.get_loaders.side_effect = Exception("mock request exception")

    with pytest.raises(NoSiibraConfigMirrorsAvailableException):
        concept.provide_registry(DummyCls)

    for connector in concept._BOOTSTRAP_CONNECTORS:
        connector.get_loaders.assert_called()


def test_provide_registry_with_connection_error():

    concept._BOOTSTRAP_CONNECTORS[0].get_loaders = MagicMock()
    concept._BOOTSTRAP_CONNECTORS[0].get_loaders.side_effect = NewConnectionError(None, "mock request error")

    with pytest.raises(NoSiibraConfigMirrorsAvailableException):
        concept.provide_registry(DummyCls)

    for connector in concept._BOOTSTRAP_CONNECTORS:
        connector.get_loaders.assert_called()
