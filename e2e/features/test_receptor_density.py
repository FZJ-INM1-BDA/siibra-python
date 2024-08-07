import pytest
import siibra

@pytest.fixture(scope="session")
def all_receptor_density_features():
    modality = siibra.modality_vocab.modality["receptor"]
    query = siibra.QueryParam(attributes=[modality])
    yield siibra.find([query], siibra.Feature)

@pytest.fixture(scope="session")
def filtered_for_fp(all_receptor_density_features):
    yield siibra.Feature.filter_facets(all_receptor_density_features, {
        "Data Type": "fingerprint"
    })
    
@pytest.fixture(scope="session")
def filtered_for_pr(all_receptor_density_features):
    yield siibra.Feature.filter_facets(all_receptor_density_features, {
        "Data Type": "cortical profile"
    })

def has_fp(feat: siibra.Feature):
    facets = feat.facets
    return len(facets.query("key == 'Data Type' & value == 'fingerprint'")) > 0

def has_profile(feat: siibra.Feature):
    facets = feat.facets
    return len(facets.query("key == 'Data Type' & value == 'cortical profile'")) > 0


# at least 1 feature
def test_all_receptor_features(all_receptor_density_features):
    assert len(all_receptor_density_features) > 0

# no duplicated ID
def test_no_duplicated_id(all_receptor_density_features):
    id_set = {feat._get(siibra.attributes.descriptions.ID).value for feat in all_receptor_density_features}
    assert len(id_set) == len(all_receptor_density_features)


# fingerprint
def test_some_has_fp(all_receptor_density_features):
    assert any(has_fp(feat) for feat in all_receptor_density_features)

def test_some_has_no_fp(all_receptor_density_features):
    assert any(not has_fp(feat) for feat in all_receptor_density_features)

def test_after_filter_all_has_fp(filtered_for_fp):
    assert all(has_fp(feat) for feat in filtered_for_fp)


# cortical profiles
def test_some_has_pr(all_receptor_density_features):
    assert any(has_profile(feat) for feat in all_receptor_density_features)

def test_some_has_no_pr(all_receptor_density_features):
    assert any(not has_profile(feat) for feat in all_receptor_density_features)

def test_after_filter_all_has_pr(filtered_for_pr):
    assert all(has_profile(feat) for feat in filtered_for_pr)
