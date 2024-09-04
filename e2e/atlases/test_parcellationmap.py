import pytest
import siibra
import nibabel as nib

ICBM_152_SPACE_ID = (
    "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2"
)

extract_regional_maps_args = [
    pytest.param(
        "difumo 128",
        "MNI 152 ICBM 2009c Nonlinear Asymmetric",
        "statistical",
        "Component 88: Lunate sulcus",
        id="full-spec-using-spc-name",
    ),
    pytest.param(
        "difumo 128",
        ICBM_152_SPACE_ID,
        "statistical",
        "Component 88: Lunate sulcus",
        id="full-spec-using-spc-id",
    ),
    pytest.param(
        "difumo 128",
        "mni 152",
        "statistical",
        "Component 88: Lunate sulcus",
        id="using-spc-name-alias",
    ),
    pytest.param(
        "difumo 128", None, "statistical", "Component 88: Lunate sulcus", id="no-space"
    ),
    pytest.param(
        "difumo 128",
        "MNI 152 ICBM 2009c Nonlinear Asymmetric",
        "statistical",
        "88",
        id="region-alias",
    ),
]


@pytest.mark.parametrize(
    "parc_spec, space_spec, map_type, region_spec", extract_regional_maps_args
)
def test_extract_region_map(parc_spec, space_spec, map_type, region_spec):
    mp = siibra.get_map(parc_spec, space_spec, map_type)
    provider = mp.extract_regional_map(region_spec)
    result = provider.get_data()
    assert isinstance(result, nib.Nifti1Image)
