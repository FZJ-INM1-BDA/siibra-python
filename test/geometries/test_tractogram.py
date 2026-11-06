"""Unit tests for Tractogram, backed by a TRX file written to a tmp_path."""

import logging

import numpy as np
import pytest
from nibabel.streamlines import Tractogram as NibTractogram

pytest.importorskip("trx", reason="trx-python requires Python >= 3.11")

from siibra.core.region import Region  # noqa: E402
from siibra.geometries import Tractogram  # noqa: E402
from siibra.geometries.providers import TrxProvider  # noqa: E402
from test.geometries.tractogram_files import make_streamlines, write_trx  # noqa: E402

STREAMLINES = make_streamlines()
GROUPS = {"Left_Arcuate": [0, 1], "Left_Arcuate_anterior": [1, 2], "Right_Uncinate": [3, 4]}
HOC1_LEFT, HOC2_LEFT = Region("hOc1 left"), Region("hOc2 left")
ROOT = Region("root", children=[HOC1_LEFT, HOC2_LEFT])
LINKS = {"Left_Arcuate": [HOC1_LEFT], "Left_Arcuate_anterior": [HOC1_LEFT, HOC2_LEFT], "Right_Uncinate": []}


@pytest.fixture
def trx_path(tmp_path):
    return write_trx(tmp_path / "bundles.trx", STREAMLINES, GROUPS)


@pytest.fixture
def tractogram(trx_path):
    return Tractogram({}, [TrxProvider(trx_path)], name="bundles", fragment_regions=LINKS)


def assert_streamlines(tractogram, indices):
    assert isinstance(tractogram, NibTractogram)
    assert len(tractogram.streamlines) == len(indices)
    for got, index in zip(tractogram.streamlines, indices):
        np.testing.assert_allclose(got, STREAMLINES[index], atol=1e-5)


def test_fetch_returns_a_nibabel_tractogram(tractogram):
    assert_streamlines(tractogram.fetch(), range(5))
    assert_streamlines(tractogram.fetch(format="tractogram"), range(5))


def test_fetch_bundle_by_exact_id(tractogram):
    assert_streamlines(tractogram.fetch(fragment="Left_Arcuate"), [0, 1])


def test_fetch_by_region_returns_shared_streamlines_once(tractogram):
    assert_streamlines(tractogram.fetch(region=HOC1_LEFT), [0, 1, 2])
    assert_streamlines(tractogram.fetch(region="hOc2 left"), [1, 2])


def test_bundles_terminating_nowhere_are_not_selected_by_region(tractogram):
    assert "Right_Uncinate" not in tractogram.get_fragments(ROOT)


def test_trx_format(tractogram):
    trx = tractogram.fetch(format="trx")
    assert sorted(trx.groups) == sorted(GROUPS)
    assert len(tractogram.fetch(fragment="Right_Uncinate", format="trx").streamlines) == 2


def test_unknown_format_raises(tractogram):
    with pytest.raises(ValueError, match="formats"):
        tractogram.fetch(format="trk")


def test_affine_is_applied_without_touching_the_cached_file(trx_path, caplog):
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    tractogram = Tractogram({}, [TrxProvider(trx_path)], affine=affine)
    np.testing.assert_allclose(
        tractogram.fetch(fragment="Left_Arcuate").streamlines[0], 2 * STREAMLINES[0], atol=1e-5
    )
    np.testing.assert_allclose(tractogram.get_coordinates()[0], 2 * STREAMLINES[0][0], atol=1e-5)
    # the stored data is unchanged
    assert_streamlines(Tractogram({}, [TrxProvider(trx_path)]).fetch(), range(5))
    with caplog.at_level(logging.WARNING, logger="siibra"):
        trx = tractogram.fetch(format="trx")
    np.testing.assert_allclose(trx.streamlines[0], STREAMLINES[0], atol=1e-5)
    assert "affine is not applied" in caplog.text


def test_boundingbox(tractogram):
    bbox = tractogram.get_boundingbox(fragment="Right_Uncinate")
    coordinates = np.concatenate(STREAMLINES[3:])
    np.testing.assert_allclose(bbox.minpoint.coordinate, coordinates.min(axis=0), atol=1e-5)
    np.testing.assert_allclose(bbox.maxpoint.coordinate, coordinates.max(axis=0), atol=1e-5)
