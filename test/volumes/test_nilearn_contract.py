"""
Pins the nilearn behaviour that `Map.extract_signals_with_nilearn` relies on.

These are not tests of siibra code. They exist because the naming and ordering of
extracted signals depend on masker attributes whose semantics are easy to get
wrong and which have changed between nilearn releases. If one of them fails after
a nilearn upgrade, `extract_signals_with_nilearn` needs revisiting.
"""

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")
maskers = pytest.importorskip("nilearn.maskers")
surface = pytest.importorskip("nilearn.surface")

N_TIMEPOINTS = 7
# 'D' is listed in the lookup table but never occurs in the data
LUT = pd.DataFrame({"index": [0, 1, 2, 3, 4], "name": ["Background", "A", "B", "C", "D"]})
EXPECTED = {0: "A", 1: "B", 2: "C"}


@pytest.fixture
def nifti_masker():
    labels = np.zeros((9, 9, 9), dtype="int16")
    labels[0:3], labels[3:6], labels[6:8] = 1, 2, 3
    data = np.random.default_rng(0).random((9, 9, 9, N_TIMEPOINTS)).astype("float32") * 1e3
    masker = maskers.NiftiLabelsMasker(
        nib.Nifti1Image(labels, np.eye(4)), lut=LUT, background_label=0, strategy="mean"
    )
    signals = np.asarray(masker.fit_transform(nib.Nifti1Image(data, np.eye(4))))
    return masker, signals


@pytest.fixture
def surface_masker():
    n = 30
    rng = np.random.default_rng(0)

    def mesh():
        return surface.InMemoryMesh(
            rng.random((n, 3)).astype("float32"),
            np.array([[i, (i + 1) % n, (i + 2) % n] for i in range(n)], dtype="int32"),
        )

    polymesh = surface.PolyMesh(left=mesh(), right=mesh())
    labels = {h: np.repeat([1, 2, 3], n // 3).astype("int32") for h in ("left", "right")}
    data = {h: rng.random((n, N_TIMEPOINTS), dtype="float32") * 1e3 for h in ("left", "right")}
    masker = maskers.SurfaceLabelsMasker(
        surface.SurfaceImage(mesh=polymesh, data=surface.PolyData(**labels)),
        lut=LUT, background_label=0, strategy="mean",
    )
    signals = np.asarray(
        masker.fit_transform(surface.SurfaceImage(mesh=polymesh, data=surface.PolyData(**data)))
    )
    return masker, signals


@pytest.mark.parametrize("fixture", ["nifti_masker", "surface_masker"])
class TestMaskerNamingContract:
    def test_region_names_maps_column_index_to_name(self, fixture, request):
        """This is what siibra uses to name the extracted columns."""
        masker, signals = request.getfixturevalue(fixture)
        assert masker.region_names_ == EXPECTED
        assert signals.shape == (N_TIMEPOINTS, len(EXPECTED))

    def test_lut_is_stored_verbatim(self, fixture, request):
        """siibra reads back the table it passed in to determine the full region set."""
        masker, _ = request.getfixturevalue(fixture)
        assert masker.lut["name"].tolist() == LUT["name"].tolist()


class TestMaskerPitfalls:
    def test_labels_includes_the_background(self, nifti_masker):
        """Why region_names_ is used for naming instead of labels_."""
        masker, signals = nifti_masker
        assert 0 in masker.labels_
        assert len(masker.labels_) != signals.shape[1]

    def test_regions_absent_from_the_data_are_dropped(self, nifti_masker):
        """Why siibra zero-fills against the lookup table after extraction."""
        masker, _ = nifti_masker
        assert "D" not in masker.region_names_.values()

    def test_maps_masker_names_columns_positionally(self):
        """Statistical maps get no names from nilearn, so siibra assigns its own."""
        rng = np.random.default_rng(0)
        maps = nib.Nifti1Image(rng.random((9, 9, 9, 4)).astype("float32"), np.eye(4))
        data = nib.Nifti1Image(rng.random((9, 9, 9, N_TIMEPOINTS)).astype("float32"), np.eye(4))
        masker = maskers.NiftiMapsMasker(maps)
        signals = np.asarray(masker.fit_transform(data))
        assert signals.shape == (N_TIMEPOINTS, 4)
        assert not hasattr(masker, "region_names_")
