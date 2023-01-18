import siibra
import pytest

concepts = [
    siibra.get_region("julich 2.9", "v2")
]

@pytest.mark.parametrize('concept', concepts)
def test_ebrains_dataset(concept):    
    features = siibra.features.get(concept, siibra.features.external.EbrainsDataset)
