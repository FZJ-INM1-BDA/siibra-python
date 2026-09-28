import numpy as np
import pytest

from siibra.volumes.volume import TimeSeriesVolume, Volume
from siibra.volumes.providers.nifti import NiftiProvider

SHAPE3D = (2, 3, 4)
LENGTH = 5


def make_volume(time=None, length=LENGTH):
    """A volume backed by an in-memory array, so no space or network is needed."""
    shape = SHAPE3D if length is None else SHAPE3D + (length,)
    data = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
    kwargs = dict(space_spec={}, providers=[NiftiProvider((data, np.eye(4)))])
    return TimeSeriesVolume(time=time, **kwargs) if time is not None else Volume(**kwargs)


class TestTimeAxis:
    def test_time_accepts_a_list(self):
        assert make_volume(time=[0, 1, 2, 3, 4]).time.tolist() == [0, 1, 2, 3, 4]

    def test_time_accepts_an_array(self):
        assert make_volume(time=np.linspace(0, 8, LENGTH)).time.tolist() == [0., 2., 4., 6., 8.]

    def test_empty_time_is_inferred_from_the_data(self):
        assert make_volume(time=[]).time.tolist() == list(range(LENGTH))

    def test_inference_rejects_a_3d_source(self):
        with pytest.raises(RuntimeError):
            make_volume(time=[], length=None).time


class TestTimeIndex:
    def test_resolves_float_time_points(self):
        assert make_volume(time=[0., .72, 1.44, 2.16, 2.88])._timeindex(1.44) == 2

    def test_rejects_unknown_time_point(self):
        with pytest.raises(ValueError):
            make_volume(time=[0, 1, 2, 3, 4])._timeindex(99)

    def test_rejects_ambiguous_time_point(self):
        with pytest.raises(ValueError):
            make_volume(time=[0, 0, 1, 2, 3])._timeindex(0)


class TestTimePointAccess:
    def test_iteration_yields_one_volume_per_time_point(self):
        vol = make_volume(time=[10., 20., 30., 40., 50.])
        timepoints = list(vol)
        assert len(timepoints) == LENGTH
        assert [v.timepoint for v in timepoints] == [10., 20., 30., 40., 50.]

    def test_fetched_time_point_is_3d(self):
        vol = make_volume(time=[0, 1, 2, 3, 4])
        img = vol.fetch(timepoint=3)
        assert img.ndim == 3
        assert img.shape == SHAPE3D

    def test_fetch_without_timepoint_returns_the_whole_series(self):
        assert make_volume(time=[0, 1, 2, 3, 4]).fetch().shape == SHAPE3D + (LENGTH,)

    def test_time_points_carry_the_expected_data(self):
        vol = make_volume(time=[0, 1, 2, 3, 4])
        whole = np.asanyarray(vol.fetch().dataobj)
        for i, v_t in enumerate(vol):
            assert np.array_equal(np.asanyarray(v_t.fetch().dataobj), whole[:, :, :, i])

    def test_get_timepoint_matches_fetch(self):
        vol = make_volume(time=[0, 1, 2, 3, 4])
        assert np.array_equal(
            np.asanyarray(vol.get_timepoint(2).fetch().dataobj),
            np.asanyarray(vol.fetch(timepoint=2).dataobj),
        )
