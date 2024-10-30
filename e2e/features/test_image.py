import siibra
import pytest
from nibabel import Nifti1Image
import numpy as np


@pytest.fixture(scope="session")
def jba29_hoc1left_cell_body_staining():
    region = siibra.get_region("2.9", "hoc1 left")
    yield siibra.find_features(region, "cell body staining")


@pytest.fixture(scope="session")
def jba303_hoc1left_cell_body_staining():
    region = siibra.get_region("3.0.3", "hoc1 left")
    yield siibra.find_features(region, "cell body staining")


@pytest.fixture(scope="session")
def bigbrain_cellbody():
    space = siibra.get_space("bigbrain")
    yield siibra.find_features(space, "cell body staining")


@pytest.fixture(scope="session")
def icbm152_t2_mri():
    space = siibra.get_space("icbm 152")
    yield siibra.find_features(space, "T2 weighted MRI")


@pytest.fixture(scope="session")
def icbm152_dti():
    space = siibra.get_space("icbm 152")
    yield siibra.find_features(space, "dti")


@pytest.fixture(scope="session")
def bigbrain_pli():
    space = siibra.get_space("bigbrain")
    yield siibra.find_features(space, "pli")


cellbody_stainging_param = [
    ("jba29_hoc1left_cell_body_staining", 45),
    ("jba303_hoc1left_cell_body_staining", 49),
    ("bigbrain_cellbody", 147),  # incl slices & reconstructed volume
    ("icbm152_t2_mri", 2),
    ("icbm152_dti", 2),
    ("bigbrain_pli", 2),
]


@pytest.mark.parametrize("fixturename, expected_len", cellbody_stainging_param)
def test_cell_body_staining(fixturename, expected_len, request):
    features = request.getfixturevalue(fixturename)
    assert (
        len(features) == expected_len
    ), f"expected {fixturename} to have {expected_len} spatial features, but got {len(features)}"


def test_pli_channel(bigbrain_pli):
    for feat in bigbrain_pli:
        for image in feat._find(siibra.attributes.datarecipes.ImageRecipe):
            img0 = image.reconfigure(channel=0)
            img1 = image.reconfigure(channel=1)
            img2 = image.reconfigure(channel=2)
            img0data, img1data, img2data = (
                img0.get_data(),
                img1.get_data(),
                img2.get_data(),
            )
            assert isinstance(img0data, Nifti1Image)
            assert isinstance(img1data, Nifti1Image)
            assert isinstance(img2data, Nifti1Image)

            assert img0data.shape == img1data.shape == img2data.shape
            assert not np.array_equal(img0data.dataobj, img1data.dataobj)
            assert not np.array_equal(img0data.dataobj, img2data.dataobj)
