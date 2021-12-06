from typing import List
from siibra.core import Parcellation
import pytest
import re

from siibra.core.parcellation import ParcellationMapModel

all_parcs = [p for p in Parcellation.REGISTRY]

def test_more_than_one_parc():
    assert len(all_parcs) > 0

@pytest.mark.parametrize('parc', all_parcs)
def test_parcs_can_be_json(parc: Parcellation):
    parc.json()

@pytest.mark.parametrize('parc', all_parcs)
def test_parcs_has_regions(parc: Parcellation):
    assert len(parc.children) > 0

all_parcs_xfail_volmes = [
    pytest.param(parc, marks=pytest.mark.xfail(reason='julich 25 has no volume')) if re.search(r'2\.5', parc.full_name) else
    pytest.param(parc, marks=pytest.mark.xfail(reason='julich 29 in big brain has no volume')) if parc.coordinate_space.get('@id') == 'minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588' else
    parc
    for parc in all_parcs
]

@pytest.mark.parametrize('parc', all_parcs_xfail_volmes)
def test_parcs_has_volumes(parc: Parcellation):
    assert len(parc.volumes) > 0
    assert all([isinstance(vol, ParcellationMapModel) for vol in parc.volumes])

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
