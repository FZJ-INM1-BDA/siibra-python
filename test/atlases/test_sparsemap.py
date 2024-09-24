from siibra.atlases.sparsemap import (
    SparseIndex,
    ReadableSparseIndex,
    WritableSparseIndex,
)
import pytest
from unittest.mock import patch, PropertyMock, MagicMock, mock_open, call
import requests
import nibabel as nib
import gzip
from pathlib import Path
import json
import numpy as np


@pytest.fixture
def mock_numpy_array():
    yield np.zeros((3, 3, 3), dtype=np.uint64)


@pytest.fixture
def session_get_mock():
    with patch.object(requests.Session, "get") as mock:
        yield mock


@pytest.fixture
def nib_frombytes_mock(mock_numpy_array):
    with patch.object(nib.Nifti1Image, "from_bytes") as mock:
        mock.return_value = nib.Nifti1Image(
            mock_numpy_array, affine=np.eye(4), dtype=np.uint64
        )
        yield mock


@pytest.fixture
def nib_load_mock(mock_numpy_array):
    with patch.object(nib, "load") as mock:
        mock.return_value = nib.Nifti1Image(
            mock_numpy_array, affine=np.eye(4), dtype=np.uint64
        )
        yield mock


@pytest.fixture
def open_mock():
    with patch(
        "builtins.open", mock_open(read_data=ReadableSparseIndex.HEADER)
    ) as mock:
        yield mock


@pytest.fixture
def gzip_decompress_mock():
    with patch.object(gzip, "decompress") as mock:
        yield mock


@pytest.fixture
def json_load_mock():
    with patch.object(json, "load") as mock:
        yield mock


mock_alias_data = {"foo": {"name": "bar"}}


@pytest.fixture
def local_readable(
    session_get_mock,
    nib_frombytes_mock,
    nib_load_mock,
    open_mock,
    gzip_decompress_mock,
    json_load_mock,
):
    yield
    session_get_mock.assert_not_called()
    nib_frombytes_mock.assert_not_called()
    nib_load_mock.assert_called_once()
    assert open_mock.call_count == 2
    gzip_decompress_mock.assert_not_called()
    json_load_mock.assert_called_once()


@pytest.fixture
def remote_readable(
    session_get_mock,
    nib_frombytes_mock,
    nib_load_mock,
    open_mock,
    gzip_decompress_mock,
    json_load_mock,
):
    yield
    session_get_mock.assert_called()
    assert session_get_mock.call_count == 3
    nib_frombytes_mock.assert_called_once()

    nib_load_mock.assert_not_called()
    open_mock.assert_not_called()
    gzip_decompress_mock.assert_called_once()
    json_load_mock.assert_not_called()


@pytest.fixture
def local_writable():
    yield


sparse_idx_args = [
    ("https://example.org/foo", "r", ReadableSparseIndex, None, "remote_readable"),
    ("https://example.org/foo", "w", None, RuntimeError, None),
    ("foo", "r", ReadableSparseIndex, None, "local_readable"),
    ("foo", "w", WritableSparseIndex, None, "local_writable"),
]


@pytest.mark.parametrize("patharg, mode, Cls, Err, fixturename", sparse_idx_args)
def test_sparseindex_init(patharg, mode, Cls, Err, fixturename, request):
    if fixturename is not None:
        fixture = request.getfixturevalue(fixturename)
    if Err:
        with pytest.raises(Err):
            SparseIndex(patharg, mode=mode)
        return
    instance = Cls(patharg, mode=mode)
    assert type(instance) is Cls


def test_remote_readable(
    remote_readable,
    gzip_decompress_mock,
    session_get_mock,
    nib_frombytes_mock,
    nib_load_mock,
    open_mock,
):
    url = "https://example.org/foo"
    gzip_decompress_mock.return_value = "gzipreturn"
    mm0 = MagicMock()
    mm1 = MagicMock()
    mm2 = MagicMock()
    session_get_mock.side_effect = [mm0, mm1, mm2]
    mm0.content = b"foo0"
    mm1.content = b"foo1"
    mm2.json.return_value = mock_alias_data

    index = SparseIndex(url, mode="r")

    session_get_mock.assert_has_calls(
        [
            call(
                url,
            ),
            call(
                url + SparseIndex.VOXEL_SUFFIX,
            ),
            call(
                url + SparseIndex.ALIAS_BBOX_SUFFIX,
            ),
        ]
    )

    gzip_decompress_mock.assert_called_once_with(mm1.content)
    nib_frombytes_mock.assert_called_once_with(gzip_decompress_mock.return_value)


def test_local_readable(
    json_load_mock,
    nib_load_mock,
    open_mock,
):
    url = "foo"
    json_load_mock.return_value = mock_alias_data
    index = SparseIndex(url, mode="r")

    nib_load_mock.assert_called_once_with(Path(url + SparseIndex.VOXEL_SUFFIX))
    assert call(Path(url), "r") in open_mock.call_args_list
    assert (
        call(Path(url + SparseIndex.ALIAS_BBOX_SUFFIX), "r") in open_mock.call_args_list
    )
