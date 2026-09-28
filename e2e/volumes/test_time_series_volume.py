from itertools import islice

import numpy as np
import nibabel
import pytest
import siibra

from siibra.volumes.volume import TimeSeriesVolume

LENGTH = 5  # each time point is compared against 128 DiFuMo maps during assignment


@pytest.fixture(scope="module")
def synthetic_vol():
    """
    A 4D volume built from the first few Julich-Brain PMaps.

    Module-scoped: building it fetches LENGTH probability maps and writes a
    ~170 MB NIfTI to the cache, which is not worth repeating per test.
    """
    julich_pmaps = siibra.get_map(
        parcellation="julich 2.9",
        space="mni152",
        maptype="statistical"
    )
    template_img = julich_pmaps.space.get_template().fetch()
    arr = np.zeros(list(template_img.shape) + [LENGTH], dtype="float32")
    for i, img in enumerate(islice(julich_pmaps.fetch_iter(), LENGTH)):
        arr[:, :, :, i] = np.asanyarray(img.dataobj)
    return siibra.volumes.from_nifti(
        nibabel.nifti1.Nifti1Image(arr, affine=template_img.affine),
        time=np.arange(LENGTH),
        space="mni152",
        name="synthetic timeseries volume"
    )


def test_timeseries_volume(synthetic_vol):
    assert isinstance(synthetic_vol, TimeSeriesVolume)
    assert len(synthetic_vol.time) == LENGTH

    timepoints = list(synthetic_vol)
    assert len(timepoints) == LENGTH
    assert [v.timepoint for v in timepoints] == list(range(LENGTH))

    img3d = synthetic_vol.fetch(timepoint=3)
    assert img3d.ndim == 3
    assert img3d.shape == synthetic_vol.fetch().shape[:3]


def test_timeseries_volume_assignment(synthetic_vol):
    difumo128 = siibra.get_map(
        parcellation="difumo 128",
        space="mni152",
        maptype="statistical"
    )
    assignments = difumo128.assign(synthetic_vol, split_components=False)

    assert not assignments.empty
    assert "time" in assignments.columns
    assert set(assignments["time"].unique()) <= set(range(LENGTH))
    assert assignments["region"].notna().all()
