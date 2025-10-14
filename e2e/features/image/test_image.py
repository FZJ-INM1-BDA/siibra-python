from typing import Union
import time
import pytest

import siibra
from siibra.core.structure import BrainStructure
from siibra.core.space import Space
from siibra.features.image.image import Image

all_image_features = [
    f
    for ft in siibra.features.Feature._SUBCLASSES[Image]
    for f in ft._get_instances()
]


@pytest.mark.parametrize("feature", all_image_features)
def test_feature_has_datasets(feature: Image):
    if feature._prerelease is True:
        pytest.xfail(
            f"Feature '{feature}' has no datasets yet since it is a prerelease data."
        )
    assert len(feature.datasets) > 0, f"{feature} has no datasets"


def test_images_datasets_names():
    start = time.time()
    all_ds_names = {ds.name for f in all_image_features for ds in f.datasets}
    end = time.time()
    duration = start - end
    assert (
        len(all_ds_names) == 19
    ), "expected 18 distinct names"  # this must be updated if new datasets are added
    assert duration < 1, "Expected getting dataset names to be less than 1s"


# Update this as new configs are added

query_and_results = [
    (siibra.spaces["bigbrain"], "CellbodyStainedSection", 145),
    (siibra.spaces["bigbrain"], "volume", 13),
    (siibra.spaces["bigbrain"], "CellBodyStainedVolumeOfInterest", 2),
    (siibra.spaces["mni152"], "volume", 5),
    (siibra.spaces["colin27"], "volume", 0),
    (siibra.get_region("julich 3.1", "hoc1 left"), "CellbodyStainedSection", 45),
    (siibra.get_region("julich 2.9", "hoc1 left"), "CellbodyStainedSection", 41),
]


@pytest.mark.parametrize("query_concept, feature_type, result_len", query_and_results)
def test_image_query_results(
    query_concept: Union[BrainStructure, Space], feature_type: str, result_len: int
):
    query_results = siibra.features.get(query_concept, feature_type)
    assert len(query_results) == result_len


def test_color_channel_fetching():
    dti_rgb_vol = [
        f
        for f in siibra.features.get(
            siibra.get_template("mni152"), siibra.features.fibres.DTIVolumeOfInterest
        )
        if "rgb" in f.name
    ][0]
    _ = dti_rgb_vol.fetch(channel=0)
    _ = dti_rgb_vol.fetch(channel=1)
    _ = dti_rgb_vol.fetch(channel=2)
