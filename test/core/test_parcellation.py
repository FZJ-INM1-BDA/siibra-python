from siibra.core import Parcellation
import pytest

all_parcs = [p for p in Parcellation.REGISTRY]

def test_more_than_one_parc():
    assert len(all_parcs) > 10

@pytest.mark.parametrize('parc', all_parcs)
def test_parcs_canbe_json(parc: Parcellation):
    parc.json()