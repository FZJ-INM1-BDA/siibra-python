import pytest
import siibra


@pytest.fixture(scope="session")
def receptor_density_search_cursor():
    modality = siibra.modality_vocab.modality["receptor"]
    query = siibra.QueryParam(attributes=[modality])
    yield siibra.SearchResult(criteria=[query], search_type=siibra.Feature)


@pytest.fixture(scope="session")
def all_receptor_density_features(receptor_density_search_cursor):
    yield receptor_density_search_cursor.find()


@pytest.fixture(scope="session")
def filtered_for_fp(receptor_density_search_cursor):
    new_cursor = receptor_density_search_cursor.reconfigure(
        spec={"category_Data Type": "fingerprint"}
    )
    yield new_cursor.find()


@pytest.fixture(scope="session")
def filtered_for_pr(receptor_density_search_cursor):
    new_cursor = receptor_density_search_cursor.reconfigure(
        spec={"category_Data Type": "cortical profile"}
    )
    yield new_cursor.find()


# at least 1 feature
def test_all_receptor_features(all_receptor_density_features):
    assert len(all_receptor_density_features) > 0


# no duplicated ID
def test_no_duplicated_id(all_receptor_density_features):
    id_set = {
        feat._get(siibra.attributes.descriptions.ID).value
        for feat in all_receptor_density_features
    }
    assert len(id_set) == len(all_receptor_density_features)


def is_fp(feat: siibra.Feature):
    for cat in feat._find(siibra.attributes.descriptions.Categorization):
        if cat.key == "Data Type" and cat.value == "fingerprint":
            return True
    return False


def is_pr(feat: siibra.Feature):
    for cat in feat._find(siibra.attributes.descriptions.Categorization):
        if cat.key == "Data Type" and cat.value == "cortical profile":
            return True
    return False


# fingerprint
def test_some_has_fp(all_receptor_density_features):
    assert any(is_fp(feat) for feat in all_receptor_density_features)


def test_some_has_no_fp(all_receptor_density_features):
    assert any(not is_fp(feat) for feat in all_receptor_density_features)


def test_after_filter_all_has_fp(filtered_for_fp):
    assert all(is_fp(feat) for feat in filtered_for_fp)


# cortical profiles
def test_some_has_pr(all_receptor_density_features):
    assert any(is_pr(feat) for feat in all_receptor_density_features)


def test_some_has_no_pr(all_receptor_density_features):
    assert any(not is_pr(feat) for feat in all_receptor_density_features)


def test_after_filter_all_has_pr(filtered_for_pr):
    assert all(is_pr(feat) for feat in filtered_for_pr)
