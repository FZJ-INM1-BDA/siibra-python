import numpy as np
import pytest
import siibra


def synthetic_timeseries(length=3):
    """A 4D volume in MNI152, built by repeating the template."""
    template_img = siibra.spaces["mni152"].get_template().fetch()
    arr = np.repeat(
        np.asanyarray(template_img.dataobj, dtype="float32")[..., None], length, axis=-1
    )
    return siibra.volumes.from_array(
        arr, template_img.affine, "mni152", name="synthetic bold", time=np.arange(length)
    )


def test_columns_follow_the_lookup_table():
    mp = siibra.get_map("julich 2.9", "mni152")
    signals = mp.extract_signals_with_nilearn(siibra.spaces["mni152"].get_template())
    assert list(signals.columns) == mp.to_BIDS_lookup_table()["name"].tolist()
    assert signals.columns.name == "region"
    assert signals.shape == (1, len(mp.to_BIDS_lookup_table()))


def test_index_is_the_time_axis_of_the_input():
    mp = siibra.get_map("julich 2.9", "mni152")
    volume = synthetic_timeseries(length=3)
    signals = mp.extract_signals_with_nilearn(volume)
    assert signals.index.name == "time"
    assert signals.index.tolist() == volume.time.tolist()


def test_statistical_maps_are_extracted_by_projection():
    mp = siibra.get_map("difumo 64", "mni152", "statistical")
    signals = mp.extract_signals_with_nilearn(siibra.spaces["mni152"].get_template())
    assert list(signals.columns) == mp.regions
    assert signals.notna().all().all()


def test_surface_map_rejects_a_volumetric_input():
    mp = siibra.get_map("julich 2.9", "fsaverage")
    with pytest.raises(ValueError):
        mp.extract_signals_with_nilearn(siibra.spaces["mni152"].get_template())


def test_volumetric_map_rejects_a_surface_input():
    mp = siibra.get_map("julich 2.9", "mni152")
    with pytest.raises(ValueError):
        mp.extract_signals_with_nilearn(siibra.spaces["fsaverage"].get_template())
