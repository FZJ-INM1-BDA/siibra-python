import siibra
import pytest
from siibra.features.anchor import AssignmentQualification
concepts = [
    (siibra.get_region("julich 2.9", "v2"), siibra.features.dataset.EbrainsDataFeature),
    (siibra.get_region("julich 2.9", "v2"), "ebrains"),
]

@pytest.mark.parametrize('concept,query_arg', concepts)
def test_ebrains_dataset(concept, query_arg):    
    features = siibra.features.get(concept, query_arg)
    assert len(features) > 0

@pytest.fixture
def features_fixture():
    return siibra.features.get(siibra.get_region("2.9", "hoc1 left"), "ebrains")

def test_each_feature(features_fixture):
    for f in features_fixture:
        for qual in f.anchor.regions.values():
            assert isinstance(qual, AssignmentQualification)
    