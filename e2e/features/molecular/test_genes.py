import pytest
from typing import Union, List
from json import JSONDecodeError
from functools import wraps
import siibra


def skip_if_allen_api_unavailable(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except JSONDecodeError:
            pytest.skip(
                f"Skipping test {test_func.__name__} due to JSONDecodeError since Allen API sent a malformed JSON"
            )
        except RuntimeError as e:
            if str(e) == "Allen institute site unavailable - please try again later.":
                pytest.skip("Skipping since Allen Institute API is unavailable.")
            else:
                raise e

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
@skip_if_allen_api_unavailable
def test_genes(parc_spec: str, region_spec: str, gene: Union[str, List[str]]):
    parc = siibra.parcellations[parc_spec]
    region = parc.get_region(region_spec)
    features = siibra.features.get(region, "gene", gene=gene)
    assert len(features) > 0, "expecting at least 1 gene feature"
    assert all(
        [isinstance(f, siibra.features.molecular.GeneExpressions) for f in features]
    ), "expecting all features to be of type GeneExpression"


@skip_if_allen_api_unavailable
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
