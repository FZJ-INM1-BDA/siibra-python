import pytest
import siibra

test_params = [("hoc1 left", "MAOA")]


@pytest.mark.parametrize("region_spec,gene", test_params)
def test_genes(region_spec: str, gene: str):
    parc = siibra.parcellations["2.9"]
    region = parc.get_region(region_spec)
    features = siibra.features.get(region, "gene", gene=gene)
    assert len(features) > 0, "expecting at least 1 gene feature"
    assert all(
        [isinstance(f, siibra.features.molecular.GeneExpressions) for f in features]
    ), "expecting all features to be of type GeneExpression"


def test_gene_exp():
    r = siibra.parcellations["2.9"].get_region("fp1 left")
    args = (r, "gene")
    kwargs = {"gene": "MAOA", "maptype": siibra.commons.MapType.STATISTICAL}
    features_higher = siibra.features.get(*args, threshold_statistical=0.9, **kwargs)
    features_lower = siibra.features.get(*args, threshold_statistical=0.2, **kwargs)

    # should have received one gene expression feature each
    assert len(features_higher) == 1
    assert len(features_lower) == 1

    # Using higher threshold should result in less gene expresssion measures
    assert len(features_lower[0].data) > len(features_higher[0].data)
