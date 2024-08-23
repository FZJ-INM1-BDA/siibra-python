from siibra.dataops.requests import EbrainsRequest, HttpRequest
from siibra.cache import CACHE

import pytest
import json
from itertools import product, repeat
from unittest.mock import PropertyMock, patch, mock_open, MagicMock
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests_mock


def test_device_flow():
    assert hasattr(EbrainsRequest, "device_flow")


http_requests_args = [("http://foo.co/bar", None, None, False, False, {})]


@pytest.mark.parametrize("url, func, msg, refresh, post, kwargs", http_requests_args)
def test_httprequests(url, func, msg, refresh, post, kwargs):
    return_filename = "return_filename"

    with patch.object(CACHE, "build_filename", return_value=return_filename) as mo:
        req = HttpRequest(url, func, msg, refresh, post, **kwargs)
        mo.assert_called_once_with(url + json.dumps(kwargs))
        assert req.cachefile == return_filename


http_test_retrieve_args = [*product([True, False], repeat=2)]


@contextmanager
def patch_all(post_flag=False, cache_flag=False):
    url = "http://foo.co/bar"
    response_text = "foo-bar"
    return_filename = "foo-bar.txt"

    m_open = mock_open()
    rename_mock = MagicMock()
    try:
        with patch("builtins.open", m_open):
            with patch("os.rename", rename_mock):
                with patch.object(
                    HttpRequest,
                    "cached",
                    new_callable=PropertyMock,
                    return_value=cache_flag,
                ) as pmock:
                    with patch.object(
                        CACHE, "build_filename", return_value=return_filename
                    ) as build_filename_mock:
                        req = HttpRequest(url, post=post_flag)
                        with requests_mock.Mocker() as req_mock:
                            if post_flag:
                                req_mock.post(url, text=response_text)
                            else:
                                req_mock.get(url, text=response_text)

                            yield req, m_open, rename_mock, pmock, build_filename_mock, response_text
    finally:
        pass


@pytest.mark.parametrize("cache_flag, post_flag", http_test_retrieve_args)
def test_httprequests_retrieve(cache_flag, post_flag):
    response_text = "foo-bar"
    return_filename = "foo-bar.txt"

    m_open = mock_open()
    rename_mock = MagicMock()
    with patch_all(post_flag=post_flag, cache_flag=cache_flag) as (
        req,
        m_open,
        rename_mock,
        pmock,
        build_filename_mock,
        response_text,
    ):
        req._retrieve()
        pmock.assert_called_once()

        if cache_flag:
            m_open.assert_not_called()
            return

        m_open.assert_called_once_with(f"{return_filename}_temp", "wb")
        handle = m_open()
        handle.write.assert_called_once_with(response_text.encode("utf-8"))
        rename_mock.assert_called_once_with(f"{return_filename}_temp", return_filename)


throttle_filename = "throttle.txt"
test_filelock_args = [
    # If first thread locks first, it finishes first
    ([throttle_filename, None], [None, 0.5], [0, 1]),
    # If second thread locks first, it finishes first
    ([None, throttle_filename], [0.5, None], [1, 0]),
]


@pytest.mark.parametrize("filenames,presleeps,expected_indices", test_filelock_args)
def test_file_lock_(filenames, presleeps, expected_indices):
    call_order = []

    def exec_retrieve(
        req: HttpRequest, overwrite_cachefile=None, presleep=None, index=None
    ):
        if overwrite_cachefile:
            req.cachefile = overwrite_cachefile
        if presleep:
            sleep(presleep)
        req._retrieve()
        call_order.append(index)
        return index

    def rename_mock_side_effect(oldname, newname):
        if newname == throttle_filename:
            sleep(1)

    with patch_all() as (
        req,
        m_open,
        rename_mock,
        pmock,
        build_filename_mock,
        response_text,
    ):
        rename_mock.side_effect = rename_mock_side_effect

        with ThreadPoolExecutor(max_workers=2) as ex:
            _ = list(
                ex.map(
                    exec_retrieve,
                    repeat(req),
                    filenames,
                    presleeps,
                    [0, 1],
                )
            )
        assert call_order[0] == expected_indices[0]
        assert call_order[1] == expected_indices[1]
