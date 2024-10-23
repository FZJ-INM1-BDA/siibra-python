from unittest.mock import mock_open, patch, MagicMock
from siibra.operations.file_fetcher import RemoteLocalDataOp
from siibra.commons.conf import PerfConf
import pytest
import requests


@pytest.fixture
def fixture_mock_open():
    mopen = mock_open()
    with patch("builtins.open", mopen):
        yield mopen


@pytest.fixture
def mock_request():
    with PerfConf.override_conf(keep_local_cache=0):
        with patch.object(requests, "get") as mock_get:
            mock_get.return_value = MagicMock()
            yield mock_get


args = [
    ("foo/bar", ("foo/bar", "rb"), None),
    ("https://example.in/buzz", None, "https://example.in/buzz"),
]


@pytest.mark.parametrize("url, mopen_calls, get_calls", args)
def test_remote_local(url, mopen_calls, get_calls, fixture_mock_open, mock_request):
    spec = RemoteLocalDataOp.generate_specs(filename=url)
    RemoteLocalDataOp().run(None, **spec)
    if mopen_calls:

        fixture_mock_open.assert_called_with(*mopen_calls)
    else:
        fixture_mock_open.assert_not_called()

    if get_calls:
        mock_request.assert_called_once_with(get_calls)
    else:
        mock_request.assert_not_called()
