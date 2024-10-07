import siibra
import pytest


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
    ("jba29_hoc1left_cell_body_staining", 43),
    ("jba303_hoc1left_cell_body_staining", 0),  # jba303 has map in big brain space
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
            image.fetch(color_channel=0)
            image.fetch(color_channel=1)
            image.fetch(color_channel=2)
