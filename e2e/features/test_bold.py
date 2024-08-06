import siibra
import pandas as pd
import pytest

@pytest.fixture
def jba29_bold_features():
    parc = siibra.get_parcellation("2.9")
    features = siibra.find_features(parc, "bold")
    yield features

def test_bold(jba29_bold_features):
    assert isinstance(jba29_bold_features, list), f"Expected bold features to be a list"
    assert len(jba29_bold_features) > 0, f"Expected at least 1 bold features"

def test_bold_facets(jba29_bold_features):
    facets = siibra.Feature.find_facets(jba29_bold_features)
    assert isinstance(facets, pd.DataFrame), f"Expecting facet to be of type dataframe"
    assert len(facets) > 0, f"Expecting at least one facet"

def test_bold_data(jba29_bold_features):
    for feat in jba29_bold_features:
        facets = feat.facets
        subjects = facets[facets["key"] == "subject"]["value"].tolist()

        for sub in subjects[:5]:
            new_f0 = feat.filter_attributes_by_facets(subject=sub)
            new_f1 = feat.filter_attributes_by_facets({"subject": sub})

            assert new_f0 == new_f1, f"different method should arrive at the same result"

            assert len(new_f0.data) > 0, f"Expected at least one data (from tabular), but got 0."
            for datum in new_f0.data:
                assert isinstance(datum, pd.DataFrame), f"Each feature.data should be a dataframe, but is insteat {type(datum)}"
