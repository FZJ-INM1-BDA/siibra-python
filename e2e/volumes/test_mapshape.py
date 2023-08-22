import pytest
import siibra
from itertools import product


# checks labelled/statistical returns volume size matches template
# see https://github.com/FZJ-INM1-BDA/siibra-python/issues/302
MNI152_ID = "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"
COLIN_ID = "minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992"

JBA_29_ID = (
    "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290"
)
JBA_30_ID = (
    "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-300"
)

HOC1_RIGHT = "Area hOc1 (V1, 17, CalcS) - right hemisphere"
FP1_RIGHT = "Area Fp1 (FPole) - right hemisphere"

STATISTIC_ENDPOINT = "statistical"
LABELLED_ENDPOINT = "labelled"

map_shape_args = product(
    ((MNI152_ID, (193, 229, 193)),),
    (JBA_29_ID,),
    (STATISTIC_ENDPOINT, LABELLED_ENDPOINT),
    (HOC1_RIGHT, FP1_RIGHT, None),
)


@pytest.mark.parametrize("space_shape,parc_id,map_endpoint,region_name", map_shape_args)
def test_map_shape(space_shape, parc_id, map_endpoint, region_name):
    if region_name is None and map_endpoint == STATISTIC_ENDPOINT:
        assert True
        return
    space_id, expected_shape = space_shape

    volume_data = None
    if region_name is not None:
        region = siibra.get_region(parc_id, region_name)
        volume_data = region.fetch_regional_map(space_id, map_endpoint)
    else:
        labelled_map = siibra.get_map(parc_id, space_id, map_endpoint)
        assert labelled_map is not None
        volume_data = labelled_map.fetch()

    assert volume_data
    assert (
        volume_data.get_fdata().shape == expected_shape
    ), f"{volume_data.get_fdata().shape}, {expected_shape}, {region_name}, {map_endpoint}, {space_id}"


def test_template_resampling():
    mp = siibra.maps.COLIN27_JBA29_LABELLED
    mp_img = mp.fetch()
    template = mp.space.get_template().fetch()
    assert mp_img.shape != template.shape
    resamp_template = mp.get_resampled_template()
    assert mp_img.shape == resamp_template.shape