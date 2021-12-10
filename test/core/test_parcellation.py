from typing import List
from siibra.core import Parcellation
import pytest
import re
from siibra.core.parcellation import BrainAtlas

from siibra.volumes.volume import VolumeSrc
from siibra.core.concept import main_openminds_registry

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
    pytest.param(parc, marks=pytest.mark.xfail(reason='julich 29 in big brain has no volume')) if parc.coordinate_space.get('@id') == 'https://openminds.ebrains.eu/instances/CoordinateSpace/a1655b99-82f1-420f-a3c2-fe80fd4c8588' else
    parc
    for parc in all_parcs
]

@pytest.mark.parametrize('parc', all_parcs_xfail_volmes)
def test_parcs_has_volumes(parc: Parcellation):
    assert len(parc.volumes) > 0
    assert all([isinstance(vol, VolumeSrc) for vol in parc.volumes])

julich_29_name = 'Julich-Brain Probabilistic Cytoarchitectonic Maps (v2.9)'
julich_29_parcs: List[Parcellation] = [parc for parc in all_parcs if parc.full_name == julich_29_name]
julich_118_name = 'Julich-Brain Probabilistic Cytoarchitectonic Maps (v1.18)'
julich_118_parcs: List[Parcellation] = [parc for parc in all_parcs if parc.full_name == julich_118_name]

def test_julich29_parcs():
    assert len(julich_29_parcs) == 6
    # big brain, mni152, colin 26
    # fsaverage, fsaverage6, hcp32k

@pytest.mark.parametrize('j29_parc', julich_29_parcs)
def test_newest(j29_parc: Parcellation):
    assert j29_parc.is_newest_version

@pytest.mark.parametrize('j118_parc', julich_118_parcs)
def test_is_not_newest(j118_parc: Parcellation):
    assert not j118_parc.is_newest_version
    

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


brainatlases = [
    item
    for item in main_openminds_registry
    if isinstance(item, BrainAtlas)    
]

def test_brain_atlases_exist():
    assert len(brainatlases) > 0

@pytest.mark.parametrize('brainatlas', brainatlases)
def test_brain_atlas_has_versions(brainatlas: BrainAtlas):
    assert len(brainatlas.has_version) > 0

@pytest.mark.parametrize('brainatlas', brainatlases)
def test_brain_atlas_has_versions_can_be_found(brainatlas: BrainAtlas):
    assert all([
        version.get("@id")
        and main_openminds_registry.provides(version.get("@id"))
        and isinstance(main_openminds_registry[version.get("@id")], Parcellation)
        for version in brainatlas.has_version
    ])
