from siibra.volumes.sparsemap import SparseIndex, SparseMap, parcellationmap
import pytest
from unittest.mock import patch, PropertyMock
from uuid import uuid4
from itertools import product


class DCls:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, __name: str):
        return self.kwargs.get(__name)


@pytest.fixture
def mock_sparse_index_load():
    with patch.object(SparseIndex, 'load') as mock:
        yield mock


@pytest.fixture
def mock_sparse_index_add_img():
    with patch.object(SparseIndex, 'add_img') as mock:
        yield mock


@pytest.fixture
def mock_sparse_map_get_region():
    with patch.object(SparseMap, 'get_region') as mock:
        yield mock


@pytest.fixture
def mock_map_fetch():
    with patch('siibra.volumes.parcellationmap.Map.fetch') as mock:
        yield mock


@pytest.fixture
def mock_sparse_index_save():
    with patch.object(SparseIndex, 'save') as mock:
        yield mock


@pytest.fixture
def mock_sparse_map_spc_parc():
    with patch.object(SparseMap, 'space', new_callable=PropertyMock) as spc_mock:
        with patch.object(SparseMap, 'parcellation', new_callable=PropertyMock) as parc_mock:
            yield (spc_mock, parc_mock)


@pytest.fixture
def sparse_map_inst():
    with patch.object(parcellationmap.Map, '__init__'):
        yield SparseMap(
            name='foo-bar',
            identifier=str(uuid4()),
            space_spec={},
            parcellation_spec={},
            indices={}
        )


@pytest.fixture
def test_sparse_idx_mocks(sparse_map_inst, mock_sparse_map_spc_parc, mock_sparse_index_load, mock_map_fetch, mock_sparse_index_save, mock_sparse_index_add_img):
    yield (sparse_map_inst, *mock_sparse_map_spc_parc, mock_sparse_index_load, mock_map_fetch, mock_sparse_index_save, mock_sparse_index_add_img)


@pytest.mark.parametrize(
    'test_sparse_idx_mocks,mem_cache_flag,disk_cached_flag',
    product(
        (None,),
        (True, False),
        (True, False),
    ),
    indirect=["test_sparse_idx_mocks"]
)
def test_sparse_index(test_sparse_idx_mocks, mem_cache_flag, disk_cached_flag):
    sparse_map_inst, spc_mock, parc_mock, mock_sparse_index_load, mock_map_fetch, mock_sparse_index_save, mock_sparse_index_add_img = test_sparse_idx_mocks
    assert isinstance(sparse_map_inst, SparseMap)

    mock_map_fetch.return_value = DCls(hello="world")

    if mem_cache_flag:
        sparse_map_inst._sparse_index_cached = DCls(foo="bar", probs=[1, 2], max=lambda: 1)
    else:
        sparse_map_inst._sparse_index_cached = None

    if disk_cached_flag:
        mock_sparse_index_load.return_value = DCls(boo="baz", probs=[1, 2], max=lambda: 1)
    else:
        mock_sparse_index_load.return_value = None
    spc_mock.return_value = DCls(id='hello world spc')
    parc_mock.return_value = DCls(id='hello world parc')

    with patch.object(SparseIndex, 'max', return_value=2):
        with patch.object(SparseMap, "__len__", return_value=3):
            with patch.object(SparseMap, 'maptype', new_callable=PropertyMock) as maptype_mock:
                sparse_map_inst.name = "map-name"
                sparse_map_inst._id = f"siibra-map-v0.0.1-map-name-{sparse_map_inst.maptype}"
                maptype_mock.return_value = "foo"

                if mem_cache_flag or disk_cached_flag:
                    _ = sparse_map_inst.sparse_index
                    if mem_cache_flag:
                        mock_sparse_index_load.assert_not_called()
                        mock_map_fetch.assert_not_called()
                        mock_sparse_index_save.assert_not_called()
                        mock_sparse_index_add_img.assert_not_called()
                        return
                    if disk_cached_flag:
                        mock_sparse_index_load.assert_called_once()
                        mock_map_fetch.assert_not_called()
                        mock_sparse_index_save.assert_not_called()
                        mock_sparse_index_add_img.assert_not_called()
                        return

                try:
                    _ = sparse_map_inst.sparse_index
                except AssertionError:
                    # TODO since new spind is created at runtime, not too sure how to bypass this assertion error
                    pass
                finally:
                    if disk_cached_flag:
                        mock_sparse_index_load.assert_called_once()
                    else:
                        mock_sparse_index_load.assert_called()
                    mock_map_fetch.assert_called()
                    mock_sparse_index_add_img.assert_called()
                    mock_sparse_index_save.assert_called_once()


def test_sparse_index_prefixes(mock_sparse_map_spc_parc, mock_sparse_index_load):
    spc_mock, parc_mock = mock_sparse_map_spc_parc
    mock_sparse_index_load.return_value = DCls(probs=[1, 2, 3], max=lambda: 2)

    spc_mock.return_value = DCls(id='hello world spc')
    parc_mock.return_value = DCls(id='hello world parc')
    foo = SparseMap(
        name='foo',
        identifier="siibra-map-v0.0.1_foo-continuous",
        space_spec=spc_mock,
        parcellation_spec=parc_mock,
        indices={}
    )
    bar = SparseMap(
        name='bar',
        identifier="siibra-map-v0.0.1_bar-continuous",
        space_spec=spc_mock,
        parcellation_spec=parc_mock,
        indices={}
    )
    with patch.object(SparseMap, 'maptype', new_callable=PropertyMock) as maptype_mock:
        maptype_mock.return_value = "foo"
        with patch.object(parcellationmap.Map, '__init__'):
            assert foo.id != bar.id
            assert foo.space is bar.space
            assert foo.parcellation is bar.parcellation
            assert foo is not bar

            foo.sparse_index
            bar.sparse_index

            call0, call1 = mock_sparse_index_load.call_args_list

            assert call0 != call1, "Prefix used should be different, based on not just space, parcellation, maptype, but also id"
            assert foo._cache_prefix != bar._cache_prefix
