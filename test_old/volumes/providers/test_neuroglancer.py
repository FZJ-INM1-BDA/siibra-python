from unittest.mock import patch, MagicMock

from siibra.volumes.providers.neuroglancer import NeuroglancerVolume
from siibra.dataops import requests


@patch.object(requests, 'HttpRequest')
def test_ngvol_info_uses_cached(MockedHttpReq):
    MockedHttpReq.return_value = MagicMock()
    MockedHttpReq.return_value.get.return_value = {"scales": []}
    vol = NeuroglancerVolume("foo/bar")
    vol._bootstrap()
    assert MockedHttpReq.called
    MockedHttpReq.return_value.get.assert_called_once()
