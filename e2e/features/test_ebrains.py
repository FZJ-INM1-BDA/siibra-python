import siibra
import pytest

@pytest.fixture
def jba29_hoc2():
    yield siibra.get_region("julich 2.9", "hoc2")

@pytest.fixture
def jba29_hoc1_lh():
    yield siibra.get_region("julich 2.9", "hoc1 left")

@pytest.mark.parametrize("fixturename", [
    "jba29_hoc2",
    "jba29_hoc1_lh",
])
def test_ebrains_features(fixturename, request):
    region = request.getfixturevalue(fixturename)
    features = siibra.find_features(region, "ebrains")
    assert len(features) > 0
