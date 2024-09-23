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
    kwargs = {"gene": "MAOA", "maptype": siibra.commons.MapType.STATISTICAL}
    features_grandparent_struct = siibra.features.get(
        *(siibra.parcellations["2.9"].get_region("frontal pole"), 'gene'), **kwargs
    )
    features_leaf_struct = siibra.features.get(
        *(siibra.parcellations["2.9"].get_region("fp1 left"), 'gene'), **kwargs
    )

    # should have received one gene expression feature each
    assert len(features_grandparent_struct) == 1
    assert len(features_leaf_struct) == 1

    # Using higher threshold should result in less gene expression measures
    assert len(features_grandparent_struct[0].data) > len(features_leaf_struct[0].data)


def test_query_w_genelist():
    genes = ["GABRA1", "GABRA2", "GABRQ"]
    p = siibra.parcellations['julich 3']
    regions = [p.get_region(spec) for spec in ["ifg 44 left", "hoc1 left"]]
    for region in regions:
        fts = siibra.features.get(region, "GeneExpressions", gene=genes)
        assert len(fts) > 0
