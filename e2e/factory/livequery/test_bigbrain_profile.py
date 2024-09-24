import siibra
import pytest
from itertools import product


@pytest.fixture(scope="session")
def big_brain_profiles():
    region = siibra.get_region("2.9", "hoc1 left")
    yield siibra.find_features(region, "Modified silver staining")


def test_no_duplicated_ptcld(big_brain_profiles):

    # TODO FIXME
    # should return one and only one profile
    # assert len(big_brain_profiles) == 1

    for feat in big_brain_profiles:
        ptclds = feat._find(siibra.attributes.locations.PointCloud)

        # Using loop to shortcurcuit and break early
        for ptclda, ptcldb in product(ptclds, repeat=2):

            # product will also pair element to itself. Skip
            if ptclda is ptcldb:
                continue

            # Test for equality. No two points should be the same
            assert ptclda != ptcldb
