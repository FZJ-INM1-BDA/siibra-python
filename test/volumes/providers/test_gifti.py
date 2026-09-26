"""Unit tests for the gifti volume providers, backed by files written to a tmp_path."""

from collections import Counter

import numpy as np
import pytest
from nibabel import gifti

from siibra.volumes.providers.gifti import (
    GiftiMesh, GiftiSurfaceLabeling, GiftiTimeSeries,
)

N_VERTICES = 12
N_TIMEPOINTS = 4
HEMISPHERES = ("left hemisphere", "right hemisphere")


def write_gii(path, arrays, intent):
    gifti.GiftiImage(
        darrays=[gifti.GiftiDataArray(a, intent=intent) for a in arrays]
    ).to_filename(str(path))
    return str(path)


def write_labels(path, labels):
    return write_gii(path, [np.asarray(labels, dtype="int32")], "NIFTI_INTENT_LABEL")


def write_timeseries(path, offset):
    arrays = [
        np.full(N_VERTICES, offset + t, dtype="float32") for t in range(N_TIMEPOINTS)
    ]
    return write_gii(path, arrays, "NIFTI_INTENT_TIME_SERIES")


def write_mesh(path, shift=0.0):
    verts = (np.arange(N_VERTICES * 3, dtype="float32").reshape(-1, 3) + shift)
    faces = np.array([[i, (i + 1) % N_VERTICES, (i + 2) % N_VERTICES]
                      for i in range(N_VERTICES)], dtype="int32")
    return write_gii(path, [verts, faces], "NIFTI_INTENT_POINTSET")


@pytest.fixture
def labelling(tmp_path):
    """Two hemispheres whose labels repeat, as configured surface maps do."""
    return GiftiSurfaceLabeling({
        "left hemisphere": write_labels(tmp_path / "lh.label.gii", [0, 1, 1, 2] * 3),
        "right hemisphere": write_labels(tmp_path / "rh.label.gii", [0, 2, 2, 1] * 3),
    })


@pytest.fixture
def timeseries(tmp_path):
    return GiftiTimeSeries({
        "Left": write_timeseries(tmp_path / "lh.func.gii", offset=0),
        "Right": write_timeseries(tmp_path / "rh.func.gii", offset=100),
    })


class TestGiftiSurfaceLabeling:
    def test_local_files_are_read_from_disk(self, labelling, tmp_path):
        for loader in labelling._loaders.values():
            assert loader.cachefile.startswith(str(tmp_path))

    def test_all_fragments_are_concatenated(self, labelling):
        labels = labelling.fetch()["labels"]
        assert labels.shape == (2 * N_VERTICES,)

    def test_fragment_selects_one_hemisphere(self, labelling):
        left = labelling.fetch(fragment="left")["labels"]
        assert left.shape == (N_VERTICES,)
        assert left.tolist() == [0, 1, 1, 2] * 3

    def test_fragment_matching_is_case_insensitive(self, labelling):
        assert np.array_equal(
            labelling.fetch(fragment="LEFT")["labels"],
            labelling.fetch(fragment="left hemisphere")["labels"],
        )

    def test_ambiguous_fragment_is_rejected(self, labelling):
        with pytest.raises(ValueError):
            labelling.fetch(fragment="hemisphere")

    def test_unknown_fragment_is_rejected(self, labelling):
        with pytest.raises(ValueError):
            labelling.fetch(fragment="cerebellum")

    def test_label_argument_returns_a_mask(self, labelling):
        mask = labelling.fetch(fragment="left", label=1)["labels"]
        assert mask.dtype == np.uint8
        assert mask.tolist() == [0, 1, 1, 0] * 3

    def test_unfragmented_source(self, tmp_path):
        prov = GiftiSurfaceLabeling(write_labels(tmp_path / "single.label.gii", [0, 1, 2, 3]))
        assert list(prov._loaders) == [None]
        assert prov.fetch()["labels"].tolist() == [0, 1, 2, 3]


class TestGiftiTimeSeries:
    def test_all_fragments_are_concatenated_per_timepoint(self, timeseries):
        series = timeseries.fetch()["timeseries"]
        assert len(series) == N_TIMEPOINTS
        assert series[0].shape == (2 * N_VERTICES,)

    def test_fragment_selects_one_hemisphere(self, timeseries):
        series = timeseries.fetch(fragment="Left")["timeseries"]
        assert len(series) == N_TIMEPOINTS
        assert series[0].shape == (N_VERTICES,)
        assert set(series[0]) == {0.0}
        assert set(series[3]) == {3.0}

    def test_fragments_keep_their_order_when_concatenated(self, timeseries):
        first = timeseries.fetch()["timeseries"][0]
        assert first[:N_VERTICES].tolist() == [0.0] * N_VERTICES      # Left
        assert first[N_VERTICES:].tolist() == [100.0] * N_VERTICES    # Right

    def test_unknown_fragment_is_rejected(self, timeseries):
        with pytest.raises(ValueError):
            timeseries.fetch(fragment="cerebellum")

    def test_each_file_is_decoded_once(self, timeseries, monkeypatch):
        """A decode per time point would mean n_timepoints x n_fragments parses."""
        calls = Counter()
        for frag, loader in timeseries._loaders.items():
            original = loader.get

            def counting_get(_frag=frag, _original=original):
                calls[_frag] += 1
                return _original()

            monkeypatch.setattr(loader, "get", counting_get)

        timeseries.fetch()
        assert calls == Counter({"Left": 1, "Right": 1})

    def test_as_polydata_names_the_hemispheres(self, timeseries):
        parts = timeseries.as_polydata().parts
        assert set(parts) == {"left", "right"}

    def test_as_polydata_rejects_unnameable_fragments(self, tmp_path):
        prov = GiftiTimeSeries({"top": write_timeseries(tmp_path / "top.func.gii", 0)})
        with pytest.raises(ValueError):
            prov.as_polydata()


class TestGiftiMesh:
    def test_fragments_are_merged_by_default(self, tmp_path):
        prov = GiftiMesh({
            "left hemisphere": write_mesh(tmp_path / "lh.surf.gii"),
            "right hemisphere": write_mesh(tmp_path / "rh.surf.gii", shift=100.0),
        })
        merged = prov.fetch()
        assert merged["verts"].shape == (2 * N_VERTICES, 3)
        assert prov.fragments == list(HEMISPHERES)

    def test_fragment_selects_one_mesh(self, tmp_path):
        prov = GiftiMesh({
            "left hemisphere": write_mesh(tmp_path / "lh.surf.gii"),
            "right hemisphere": write_mesh(tmp_path / "rh.surf.gii", shift=100.0),
        })
        assert prov.fetch(fragment="left")["verts"].shape == (N_VERTICES, 3)

    def test_fetch_iter_yields_one_mesh_per_fragment(self, tmp_path):
        prov = GiftiMesh({
            "left hemisphere": write_mesh(tmp_path / "lh.surf.gii"),
            "right hemisphere": write_mesh(tmp_path / "rh.surf.gii", shift=100.0),
        })
        assert [m["verts"].shape for m in prov.fetch_iter()] == [(N_VERTICES, 3)] * 2

    def test_boundingbox_spans_the_vertices(self, tmp_path):
        prov = GiftiMesh(write_mesh(tmp_path / "single.surf.gii"))
        bbox = prov.get_boundingbox()
        verts = prov.fetch()["verts"]
        assert np.allclose(bbox.minpoint, verts.min(0))
        assert np.allclose(bbox.maxpoint, verts.max(0))

    def test_unsupported_fetch_arguments_are_rejected(self, tmp_path):
        prov = GiftiMesh(write_mesh(tmp_path / "single.surf.gii"))
        with pytest.raises(NotImplementedError):
            prov.fetch(resolution_mm=1.0)
