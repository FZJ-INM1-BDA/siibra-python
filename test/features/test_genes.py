import pytest
import siibra

test_params = [
    ("hoc1 left", "MAOA")
]


@pytest.mark.parametrize("region_spec,gene", test_params)
def test_genes(region_spec: str, gene: str):
    parc = siibra.parcellations['2.9']
    region = parc.get_region(region_spec)
    features = siibra.features.get(region, "gene", gene=gene)
    assert len(features) > 0, "expecting at least 1 gene feature"
    assert all([
        isinstance(f, siibra.features.molecular.GeneExpressions) for f in features
    ]), "expecting all features to be of type GeneExpression"
