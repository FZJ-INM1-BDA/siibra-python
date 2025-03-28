import pytest
from typing import Union, List
from json import JSONDecodeError
from requests import RequestException
from functools import wraps
import siibra


def xfail_if_allen_api_unavailable(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except JSONDecodeError:
            pytest.xfail(
                f"Skipping test {test_func.__name__} due to JSONDecodeError since Allen API sent a malformed JSON"
            )
        except RuntimeError as e:
            if str(e) == "Allen institute site unavailable - please try again later.":
                pytest.xfail("Skipping since Allen Institute API is unavailable.")
            else:
                raise e
        except siibra.livequeries.allen.InvalidAllenAPIResponseException as e:
            pytest.xfail(
                f"Skipping test {test_func.__name__} due to invalid response from Allen API:\n{e}"
            )
        except (RequestException, siibra.retrieval.requests.SiibraHttpRequestError) as e:
            pytest.xfail(
                f"Skipping test {test_func.__name__} due to request error from Allen API:\n{e}"
            )

    return wrapper


test_params = [
    ("julich 2.9", "hoc1 left", "MAOA"),
    ("julich 3.0.3", "hoc1 left", "MAOA"),
    ("julich 3.1", "hoc1 left", "MAOA"),
    ("julich 3.1", "ifg 44 left", "MAOA"),
    ("julich 3.1", "hoc1 left", ["GABRA1", "GABRA2", "GABRQ"]),
    ("julich 3.1", "ifg 44 left", ["GABRA1", "GABRA2", "GABRQ"]),
]


@pytest.mark.parametrize("parc_spec, region_spec, gene", test_params)
@xfail_if_allen_api_unavailable
def test_genes(parc_spec: str, region_spec: str, gene: Union[str, List[str]]):
    parc = siibra.parcellations[parc_spec]
    region = parc.get_region(region_spec)
    features = siibra.features.get(region, "gene", gene=gene)
    assert len(features) > 0, "expecting at least 1 gene feature"
    assert all(
        [isinstance(f, siibra.features.molecular.GeneExpressions) for f in features]
    ), "expecting all features to be of type GeneExpression"


@xfail_if_allen_api_unavailable
def test_gene_exp_w_parent_structures():
    kwargs = {"gene": "MAOA"}
    features_grandparent_struct = siibra.features.get(
        *(siibra.parcellations["2.9"].get_region("frontal pole"), "gene"), **kwargs
    )
    features_leaf_struct = siibra.features.get(
        *(siibra.parcellations["2.9"].get_region("fp1 left"), "gene"), **kwargs
    )

    # must find a gene expression for both
    assert len(features_grandparent_struct) == 1
    assert len(features_leaf_struct) == 1

    # grandparent area should contain more measurements
    assert len(features_grandparent_struct[0].data) > len(features_leaf_struct[0].data)


@xfail_if_allen_api_unavailable
def test_no_probes_found_in_concept():
    bbox = siibra.locations.BoundingBox(
        [-75, -110, -75],
        [-74, -109, -74],
        space='mni152',
    )
    features = siibra.features.get(
        bbox,
        siibra.features.molecular.GeneExpressions,
        gene=siibra.vocabularies.GENE_NAMES.G0S2,
    )
    assert features == []


@xfail_if_allen_api_unavailable
def test_reproducing_query_results():
    region = siibra.get_region("julich 2.9", "V1")
    features = siibra.features.get(
        region, siibra.features.molecular.GeneExpressions,
        gene=siibra.vocabularies.GENE_NAMES.GABARAPL2
    )
    assert len(features) == 1
    gene_exp = features[0]
    assert set(gene_exp.data['gene']) == {'GABARAPL2'}
    assert gene_exp.data.iloc[0]['level'] == 10.1143
    assert gene_exp.data.iloc[2]['zscore'] == -0.6653
    assert gene_exp.data.iloc[3]['mni_xyz'] == (-6.0, -86.0, 10.0)
    assert gene_exp.data.iloc[4]['probe_id'] == 1046316
    assert gene_exp.data.iloc[4]['donor_name'] == "H0351.1015"
