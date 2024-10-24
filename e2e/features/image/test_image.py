import siibra
import pytest
from siibra.features.image.image import Image
import time

PRERELEASE_FEATURES_W_NO_DATASET = [
    "The Enriched Connectome - Block face images of full sagittal human brain sections (blockface)",
    "The Enriched Connectome - 3D polarized light imaging connectivity data of full sagittal human brain sections (HSV fibre orientation map)",
]
all_image_features = [f for ft in siibra.features.Feature._SUBCLASSES[siibra.features.image.image.Image] for f in ft._get_instances()]


@pytest.mark.parametrize("feature", all_image_features)
def test_feature_has_datasets(feature: Image):
    if feature.name in PRERELEASE_FEATURES_W_NO_DATASET:
        if len(feature.datasets) > 0:
            pytest.fail(f"Feature '{feature}' was listed as prerelase previosly but now have dataset information. Please update `PRERELEASE_FEATURES_W_NO_DATASET`")
        pytest.skip(f"Feature '{feature}' has no datasets yet as it is a prerelease data.")
    assert len(feature.datasets) > 0, f"{feature} has no datasets"


def test_images_datasets_names():
    start = time.time()
    all_ds_names = {ds.name for f in all_image_features for ds in f.datasets}
    end = time.time()
    duration = start - end
    assert len(all_ds_names) == 10, "expected 10 distinct names"  # this must be updated if new datasets are added
    assert duration < 1, "Expected getting dataset names to be less than 1s"


# Update this as new configs are added
query_and_results = [
    (siibra.features.get(siibra.get_template("big brain"), "CellbodyStainedSection"), 145),
    (siibra.features.get(siibra.get_template("big brain"), "CellBodyStainedVolumeOfInterest"), 2),
    (siibra.features.get(siibra.get_template("mni152"), "image", restrict_space=True), 4),
    (siibra.features.get(siibra.get_template("mni152"), "image", restrict_space=False), 161),
    (siibra.features.get(siibra.get_region('julich 3.1', 'hoc1 left'), "CellbodyStainedSection"), 45),
    (siibra.features.get(siibra.get_region('julich 2.9', 'hoc1 left'), "CellbodyStainedSection"), 41)
]


@pytest.mark.parametrize("query_results, result_len", query_and_results)
def test_image_query_results(
    query_results: Image,
    result_len: int
):
    assert len(query_results) == result_len


def test_color_channel_fetching():
    dti_rgb_vol = [
        f
        for f in siibra.features.get(
            siibra.get_template('mni152'),
            siibra.features.fibres.DTIVolumeOfInterest
        )
        if 'rgb' in f.name
    ][0]
    _ = dti_rgb_vol.fetch(channel=0)
    _ = dti_rgb_vol.fetch(channel=1)
    _ = dti_rgb_vol.fetch(channel=2)
