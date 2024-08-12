from siibra.atlases.sparsemap import SparseIndex, ReadableSparseIndex, WritableSparseIndex
import pytest
from unittest.mock import patch, PropertyMock, MagicMock, mock_open, call
import requests
import nibabel as nib
import gzip
from pathlib import Path


@pytest.fixture
def session_get_mock():
    with patch.object(requests.Session, "get") as mock:
        yield mock


@pytest.fixture
def nib_frombytes_mock():
    with patch.object(nib.Nifti1Image, "from_bytes") as mock:
        yield mock


@pytest.fixture
def nib_load_mock():
    with patch.object(nib, "load") as mock:
        yield mock


@pytest.fixture
def open_mock():
    with patch("builtins.open", mock_open(read_data="foobar")) as mock:
        yield mock


@pytest.fixture
def gzip_decompress():
    with patch.object(gzip, "decompress") as mock:
        yield mock


sparse_idx_args = [
    ("https://example.org/foo", "r", ReadableSparseIndex, None),
    ("https://example.org/foo", "w", None, RuntimeError),
    ("foo", "r", ReadableSparseIndex, None),
    ("foo", "w", WritableSparseIndex, None)
]


@pytest.mark.parametrize("patharg, mode, Cls, Err", sparse_idx_args)
def test_sparseindex_init(patharg, mode, Cls, Err, session_get_mock, nib_frombytes_mock, nib_load_mock, open_mock, gzip_decompress):
    if Err:
        with pytest.raises(Err):
            SparseIndex(patharg, mode=mode)
        return
    instance = Cls(patharg, mode=mode)
    assert type(instance) is Cls


def test_remote_readable(session_get_mock, gzip_decompress, nib_frombytes_mock, nib_load_mock, open_mock):
    url = "https://example.org/foo"
    gzip_decompress.return_value = "gzipreturn"
    mm0 = MagicMock()
    mm1 = MagicMock()
    session_get_mock.side_effect = [mm0, mm1, mm0, mm1]
    mm0.content = b"foo0"
    mm1.content = b"foo1"

    index = SparseIndex(url, mode="r")

    session_get_mock.assert_has_calls([
        call(url + SparseIndex.VOXEL_SUFFIX,),
        call(url + SparseIndex.META_SUFFIX,),
    ])

    gzip_decompress.assert_called_once_with(mm0.content)
    nib_frombytes_mock.assert_called_once_with(gzip_decompress.return_value)

    nib_load_mock.assert_not_called()
    open_mock.assert_not_called()


def test_local_readable(session_get_mock, gzip_decompress, nib_frombytes_mock, nib_load_mock, open_mock):
    url = "foo"

    index = SparseIndex(url, mode="r")

    session_get_mock.assert_not_called()

    gzip_decompress.assert_not_called()
    nib_frombytes_mock.assert_not_called()

    nib_load_mock.assert_called_once_with(Path(url + SparseIndex.VOXEL_SUFFIX))
    open_mock.assert_called_once_with(Path(url + SparseIndex.META_SUFFIX), "r")

