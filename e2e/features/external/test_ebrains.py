import siibra
import pytest

concepts = [
    (siibra.get_region("julich 2.9", "v2"), siibra.features.external.EbrainsDataset),
    (siibra.get_region("julich 2.9", "v2"), "ebrains"),
]

@pytest.mark.parametrize('concept,query_arg', concepts)
def test_ebrains_dataset(concept, query_arg):    
    features = siibra.features.get(concept, query_arg)
    assert len(features) > 0
    for feature in features:
        print(feature.description)
