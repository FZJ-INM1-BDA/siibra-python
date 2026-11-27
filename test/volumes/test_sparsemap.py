from siibra.volumes.sparsemap import SparseIndex, SparseMap, parcellationmap
from siibra.locations.pointcloud import PointCloud
import numpy as np
import pytest
from unittest.mock import patch, PropertyMock, MagicMock
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


def _build_sparse_map_with_two_regions():
    arr0 = np.zeros((6, 6, 6), dtype="float32")
    arr0[1, 1, 1] = 0.7
    arr1 = np.zeros((6, 6, 6), dtype="float32")
    arr1[4, 4, 4] = 0.9

    spind = SparseIndex()
    spind.add_img(arr0, np.eye(4))
    spind.add_img(arr1, np.eye(4))

    mp = SparseMap(
        identifier=str(uuid4()),
        name="sparse-read-voxel-test",
        space_spec={},
        parcellation_spec={},
        indices={"region 0": [{"volume": 0}], "region 1": [{"volume": 1}]},
        volumes=[DCls(formats={"nii"}, provides_image=True) for _ in range(2)],
    )
    mp._sparse_index_cached = spind
    return mp


def test_read_voxel_ignores_unmapped_in_bounds_voxel():
    mp = _build_sparse_map_with_two_regions()
    # (0, 0, 0) is in bounds but unmapped (voxels[0, 0, 0] == -1);
    # it must not wrap around to spind.probs[-1].
    assert mp._read_voxel(np.array([0]), np.array([0]), np.array([0])) == []


def test_read_voxel_skips_unmapped_and_out_of_bounds_points():
    mp = _build_sparse_map_with_two_regions()
    # index 0: unmapped (-1 in spind.voxels), index 1: out of bounds (negative),
    # index 2: out of bounds (>= shape), index 3: mapped to volume 0
    X = np.array([0, -1, 10, 1])
    Y = np.array([0, 0, 0, 1])
    Z = np.array([0, 0, 0, 1])

    result = mp._read_voxel(X, Y, Z)
    assert len(result) == 1
    pointindex, volume, fragment, value = result[0]
    assert pointindex == 3
    assert volume == 0
    assert fragment is None
    assert value == pytest.approx(0.7)


def test_read_voxel_accepts_scalar_coordinates():
    mp = _build_sparse_map_with_two_regions()

    mapped = mp._read_voxel(1, 1, 1)
    assert len(mapped) == 1
    assert mapped[0][:3] == (None, 0, None)
    assert mapped[0][3] == pytest.approx(0.7)

    assert mp._read_voxel(0, 0, 0) == []
    assert mp._read_voxel(-1, 0, 0) == []
    assert mp._read_voxel(6, 0, 0) == []


def test_assign_pointcloud_does_not_assign_unmapped_or_out_of_bounds_points():
    mp = _build_sparse_map_with_two_regions()
    mock_space = MagicMock()
    points = PointCloud([(0, 0, 0), (-5, 0, 0), (1, 1, 1), (20, 20, 20)], space=None, sigma_mm=0)

    with patch.object(SparseMap, "space", new_callable=PropertyMock, return_value=mock_space), \
            patch.object(PointCloud, "space", new_callable=PropertyMock, return_value=mock_space), \
            patch.object(PointCloud, "warp", lambda pts, *args, **kwargs: pts), \
            patch.object(SparseMap, "get_region", side_effect=lambda index=None, **kw: f"region {index.volume}"):
        df = mp.assign(points)

    assert len(df) == 1
    assert df.iloc[0]["input structure"] == 2
    assert df.iloc[0]["region"] == "region 0"
    assert df.iloc[0]["map value"] == pytest.approx(0.7)
