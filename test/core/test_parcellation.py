from typing import List
from siibra.core import Parcellation
import pytest

all_parcs = [p for p in Parcellation.REGISTRY]

def test_more_than_one_parc():
    assert len(all_parcs) > 0

@pytest.mark.parametrize('parc', all_parcs)
def test_parcs_can_be_json(parc: Parcellation):
    parc.json()

@pytest.mark.parametrize('parc', all_parcs)
def test_parcs_has_regions(parc: Parcellation):
    assert len(parc.children) > 0

julich_29_name = 'Julich-Brain Probabilistic Cytoarchitectonic Maps (v2.9)'
julich_29_parcs: List[Parcellation] = [parc for parc in all_parcs if parc.full_name == julich_29_name]

def test_julich29_parcs():
    assert len(julich_29_parcs) == 6
    # big brain, mni152, colin 26
    # fsaverage, fsaverage6, hcp32k

# get all combination of indices
pairwise = [
    (i, j)
    for i, _ in enumerate(julich_29_parcs)
    for j, _ in enumerate(julich_29_parcs)
    if i < j
]

@pytest.mark.parametrize('idx_1,idx_2', pairwise)
def test_parcs_get_region(idx_1,idx_2):
    assert julich_29_parcs[idx_1].children[0] is not julich_29_parcs[idx_2].children[0]
    
    # changing name attr does not affect other parcellations
    idx_1_name = julich_29_parcs[idx_1].children[0].name
    assert julich_29_parcs[idx_1].children[0].name == julich_29_parcs[idx_2].children[0].name
    julich_29_parcs[idx_1].children[0].name = 'foo-bar'
    assert julich_29_parcs[idx_1].children[0].name != julich_29_parcs[idx_2].children[0].name
    julich_29_parcs[idx_1].children[0].name = idx_1_name
    assert julich_29_parcs[idx_1].children[0].name == julich_29_parcs[idx_2].children[0].name
