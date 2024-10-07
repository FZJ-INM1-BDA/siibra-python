import pytest
import siibra

test_params = [("julich 2.9", "hoc1 left", "MAOA")]


@pytest.mark.parametrize("parc_spec, region_spec, gene", test_params)
def test_genes(parc_spec, region_spec, gene):
    region = siibra.get_region(parc_spec, region_spec)
    features = siibra.find_features(region, "gene", genes=[gene])
    assert len(features) > 0, f"Expected at least 1 feature, but got 0."


def test_gene_exp():
    kwargs = {"genes": ["MAOA"]}

    features_grandparent_struct = siibra.find_features(
        siibra.get_region("2.9", "frontal pole"), "gene", **kwargs
    )
    features_leaf_struct = siibra.find_features(
        siibra.get_region("2.9", "fp1 left"), "gene", **kwargs
    )

    # should have received one gene expression feature each
    assert len(features_grandparent_struct) == 1
    assert len(features_leaf_struct) == 1

    # Using higher threshold should result in less gene expresssion measures
    assert len(features_grandparent_struct[0].data[0]) > len(
        features_leaf_struct[0].data[0]
    )


def test_query_w_genelist():
    genes = ["GABRA1", "GABRA2", "GABRQ"]
    p = siibra.parcellations["julich 3.0.3"]
    regions = [p.get_region(spec) for spec in ["ifg 44 left", "hoc1 left"]]
    for region in regions:
        fts = siibra.find_features(region, "gene", gene=genes)
        assert len(fts) > 0
