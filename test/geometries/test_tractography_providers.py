"""Unit tests for the TRX and TRK geometry providers, backed by files written to a tmp_path."""

import os

import numpy as np
import pytest

pytest.importorskip("trx", reason="trx-python requires Python >= 3.11")

from siibra.geometries.providers import TrkProvider, TrxProvider  # noqa: E402
from test.geometries.tractogram_files import make_streamlines, write_trk, write_trx  # noqa: E402

STREAMLINES = make_streamlines()
GROUPS = {"Left_Arcuate": [0, 1], "Left_Arcuate_anterior": [1, 2], "Right_Uncinate": [3, 4]}


@pytest.fixture
def trx_provider(tmp_path):
    return TrxProvider(write_trx(tmp_path / "bundles.trx", STREAMLINES, GROUPS))


def assert_streamlines_equal(actual, expected_indices):
    assert len(actual) == len(expected_indices)
    for got, index in zip(actual, expected_indices):
        np.testing.assert_allclose(got, STREAMLINES[index], atol=1e-5)


def test_trx_groups_are_fragments_in_sorted_order(trx_provider):
    assert trx_provider.fragments == sorted(GROUPS)


def test_trx_fetch_everything(trx_provider):
    assert_streamlines_equal(trx_provider.fetch().streamlines, range(len(STREAMLINES)))


def test_trx_fetch_groups_without_duplicates(trx_provider):
    selection = trx_provider.fetch(["Left_Arcuate", "Left_Arcuate_anterior"])
    assert_streamlines_equal(selection.streamlines, [0, 1, 2])


def test_trx_selection_shares_the_memory_map(trx_provider):
    selection = trx_provider.fetch(["Right_Uncinate"])
    assert np.shares_memory(selection.streamlines._data, trx_provider.fetch().streamlines._data)


def test_trx_unknown_group_raises(trx_provider):
    with pytest.raises(ValueError, match="No group"):
        trx_provider.fetch(["Corpus_Callosum"])


def test_trx_coordinates(trx_provider):
    all_coordinates = trx_provider.get_coordinates()
    assert isinstance(all_coordinates, np.memmap)
    np.testing.assert_allclose(all_coordinates, np.concatenate(STREAMLINES), atol=1e-5)
    np.testing.assert_allclose(
        trx_provider.get_coordinates(["Right_Uncinate"]),
        np.concatenate(STREAMLINES[3:]),
        atol=1e-5,
    )


def test_trx_takes_a_single_source():
    with pytest.raises(ValueError, match="groups inside the TRX"):
        TrxProvider({"a": "a.trx"})


@pytest.fixture
def trk_sources(tmp_path):
    return {
        "Left_Arcuate": write_trk(tmp_path / "left_arcuate.trk", STREAMLINES[:2]),
        "Right_Uncinate": write_trk(tmp_path / "right_uncinate.trk", STREAMLINES[2:]),
    }


def test_trk_files_become_groups(isolated_cache, trk_sources):
    provider = TrkProvider(trk_sources)
    assert provider.fragments == ["Left_Arcuate", "Right_Uncinate"]
    assert_streamlines_equal(provider.fetch().streamlines, range(5))
    assert_streamlines_equal(provider.fetch(["Right_Uncinate"]).streamlines, [2, 3, 4])


def test_trk_is_converted_once_into_the_cache(isolated_cache, trk_sources, monkeypatch):
    path = TrkProvider(trk_sources)._trx_path()
    assert os.path.dirname(path) == str(isolated_cache) and path.endswith(".trx")

    def fail(self, target):
        raise AssertionError("converted again")

    monkeypatch.setattr(TrkProvider, "_convert", fail)
    assert TrkProvider(trk_sources)._trx_path() == path
    assert sorted(os.listdir(isolated_cache)) == sorted([os.path.basename(path), os.path.basename(path) + ".lock"])


def test_trk_conversion_follows_changes_of_local_files(isolated_cache, trk_sources):
    path = TrkProvider(trk_sources)._trx_path()
    stat = os.stat(trk_sources["Left_Arcuate"])
    os.utime(trk_sources["Left_Arcuate"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert TrkProvider(trk_sources)._trx_path() != path


def test_single_trk_has_no_fragments(isolated_cache, tmp_path):
    provider = TrkProvider(write_trk(tmp_path / "all.trk", STREAMLINES))
    assert provider.fragments == []
    assert len(provider.fetch().streamlines) == len(STREAMLINES)


def test_missing_trx_python_is_explained(trx_provider, monkeypatch):
    from siibra.geometries.providers import tractography
    monkeypatch.setattr(tractography, "_trx", None)
    with pytest.raises(ImportError, match="Python >= 3.11"):
        trx_provider.fetch()
