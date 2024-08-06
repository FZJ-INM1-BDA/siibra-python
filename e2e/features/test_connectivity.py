import siibra
import pandas as pd
import pytest
from zipfile import ZipFile


@pytest.fixture
def conn_modalities():
    yield siibra.modality_vocab.modality.find("connectivity")

@pytest.fixture
def jba3_conn_features(conn_modalities):
    
    parc = siibra.get_parcellation("julich 3.0.3")
    yield [
        feat
        for mod in conn_modalities
        for feat in siibra.find_features(parc, mod)
    ]

def test_conn_features(jba3_conn_features):
    assert isinstance(jba3_conn_features, list), f"Expected conn features to be a list"
    assert len(jba3_conn_features) > 0, f"Expected at least 1 conn features"


def test_conn_facets(jba3_conn_features):
    facets = siibra.Feature.find_facets(jba3_conn_features)
    assert isinstance(facets, pd.DataFrame), f"Expecting facet to be of type dataframe"
    assert len(facets) > 0, f"Expecting at least one facet"


def test_breaking_str_regional_connectivity():
    """
    This test validates the breaking change that 'RegionalConnectivity' can no longer be used
    to find all features related to connectivity.
    """
    parc = siibra.get_parcellation("julich 3.0.3")
    with pytest.raises(IndexError):
        siibra.find_features(parc, "RegionalConnectivity")


def test_breaking_connectivity():
    """
    This test validates the breaking change that using 'connectivity' as search string 
    yields very different result. 'connectivity' resolves to TracerConnectivity
    """
    parc = siibra.get_parcellation("julich 2.9")
    features = siibra.find_features(parc, "connectivity")
    assert len(features) == 0


args = [
    ("julich 2.9", "StreamlineCounts"),
    ("julich 3.0.3", "Anatomo"),
    ("julich 2.9", siibra.modality_vocab.modality.STREAMLINECOUNTS),
    ("julich 2.9", siibra.modality_vocab.modality.STREAMLINELENGTHS),
    ("julich 2.9", siibra.modality_vocab.modality["streamline counts"]),
    ("julich 2.9", siibra.modality_vocab.modality["streamline lengths"]),
]


@pytest.mark.parametrize("parc_spec, query_arg", args)
def test_get_connectivity(parc_spec, query_arg):
    parc = siibra.get_parcellation(parc_spec)
    features = siibra.find_features(parc, query_arg)
    assert len(features) > 0, f"Expected at least one feature with modality {query_arg} at {parc_spec}"

def test_to_zip(jba3_conn_features):
    for feat in jba3_conn_features[:3]:
        facets = feat.facets
        subjects = facets[facets["key"] == "subject"]["value"].tolist()
        for sub in subjects[:3]:
            new_feat = feat.filter_attributes_by_facets(subject=sub)
            new_feat.to_zip("test.zip")
            zf = ZipFile("test.zip")
            all_filenames = [f.filename for f in zf.filelist]
            assert len([f for f in all_filenames if f.endswith(".csv")]) > 0, "Expected at least 1 csv file, got 0"
            assert len([f for f in all_filenames if  f.endswith("README.md")]) > 0, "Expected readme file"
