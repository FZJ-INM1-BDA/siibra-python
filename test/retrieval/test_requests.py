from siibra.retrieval.requests import EbrainsRequest, HttpRequest, CACHE, SiibraHttpRequestError, _get_suffix

import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep
from unittest.mock import patch

import pytest

import requests_mock


def test_device_flow():
    assert hasattr(EbrainsRequest, "device_flow")


http_requests_args = [("http://foo.co/bar", None, None, False, False, {})]


@pytest.mark.parametrize("url, func, msg, refresh, post, kwargs", http_requests_args)
def test_httprequests(url, func, msg, refresh, post, kwargs):
    return_filename = "return_filename"

    with patch.object(CACHE, "build_filename", return_value=return_filename) as mo:
        req = HttpRequest(url, func, msg, refresh, post, **kwargs)
        mo.assert_called_once_with(
            url + json.dumps(kwargs),
            suffix=_get_suffix(url),
        )
        assert req.cachefile == return_filename


URL = "http://foo.co/bar"


@pytest.fixture
def cachefile(tmp_path):
    """Point HttpRequest's cache file into a temporary folder."""
    path = str(tmp_path / "foo-bar.txt")
    with patch.object(CACHE, "build_filename", return_value=path):
        yield path


@pytest.mark.parametrize("post_flag", [True, False])
def test_httprequests_retrieve(cachefile, post_flag):
    with requests_mock.Mocker() as m:
        m.register_uri("POST" if post_flag else "GET", URL, text="foo-bar")
        req = HttpRequest(URL, post=post_flag)
        req._retrieve()
        assert open(cachefile).read() == "foo-bar"
        assert m.call_count == 1

        HttpRequest(URL, post=post_flag)._retrieve()  # cached: no second request
        assert m.call_count == 1


def test_httprequests_retrieve_refresh(cachefile):
    with open(cachefile, "w") as f:
        f.write("old")
    with requests_mock.Mocker() as m:
        m.get(URL, text="new")
        req = HttpRequest(URL, refresh=True)
        req._retrieve()
        assert open(cachefile).read() == "new"
        assert req.refresh is False
        assert m.call_count == 1


def test_httprequests_concurrent_download_once(cachefile):
    def slow_response(request, context):
        sleep(0.2)  # widen the race window
        return "foo-bar"

    n = 8
    barrier = Barrier(n)

    def retrieve(_):
        barrier.wait()
        HttpRequest(URL)._retrieve()
        return open(cachefile).read()

    with requests_mock.Mocker() as m:
        m.get(URL, text=slow_response)
        with ThreadPoolExecutor(n) as ex:
            results = list(ex.map(retrieve, range(n)))
        assert m.call_count == 1, "the file should be downloaded only once"
    assert results == ["foo-bar"] * n


def test_httprequests_failed_download_leaves_no_file(cachefile):
    folder = os.path.dirname(cachefile)
    with requests_mock.Mocker() as m:
        m.get(URL, status_code=503)
        with pytest.raises(SiibraHttpRequestError):
            HttpRequest(URL)._retrieve()
    assert not os.path.exists(cachefile)
    assert [f for f in os.listdir(folder) if ".tmp-" in f] == []


def test_zipfile_member_extracted_once(tmp_path, monkeypatch):
    from zipfile import ZipFile
    from siibra.retrieval import requests as requests_module
    from siibra.retrieval.requests import Decoder, ZipfileRequest

    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("folder/member.txt", "member content")
    monkeypatch.setattr(CACHE, "folder", str(tmp_path / "cache"))
    os.makedirs(CACHE.folder)

    copies = []
    real_copy = requests_module.shutil.copyfileobj

    def slow_copy(src, dst, *args, **kwargs):
        copies.append(1)
        sleep(0.2)  # widen the race window
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(requests_module.shutil, "copyfileobj", slow_copy)
    read_file = Decoder(lambda path: open(path).read(), input_type="FILE")
    n = 8
    barrier = Barrier(n)

    def get(_):
        barrier.wait()
        return ZipfileRequest(str(archive), "folder/member.txt", func=read_file).get()

    with ThreadPoolExecutor(n) as ex:
        results = list(ex.map(get, range(n)))
    assert results == ["member content"] * n
    assert len(copies) == 1, "the member should be extracted only once"
