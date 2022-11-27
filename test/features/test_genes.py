import pytest
import siibra
from siibra.features.genes import GeneExpression

test_params = [
    ("hoc1 left", "MAOA")
]

@pytest.mark.parametrize("region_spec,gene", test_params)
def test_genes(region_spec:str, gene: str):
    parc = siibra.parcellations['2.9']
    region = parc.get_region(region_spec)
    features = siibra.get_features(region, "gene", gene=gene)
    assert len(features) > 0, f"expecting at least 1 gene feature"
    assert all([
        isinstance(f, GeneExpression) for f in features
    ]), f"expecting all features to be of type GeneExpression"
    assert all([
        hasattr(f, 'structure') and hasattr(f, "top_level_structure") for f in features
    ]), f"expecting all features to have structure and top_level_structure attributes"
