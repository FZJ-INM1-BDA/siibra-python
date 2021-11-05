from siibra.core.json_encoder import JSONEncoder
from typing import List
import siibra
from siibra.core import Atlas, Parcellation
from siibra.features.receptors import ReceptorDistribution, DensityProfile, DensityFingerprint
import pytest
density_profile_param=[
    ('human', '2 9', 'hoc1 left')
]

@pytest.mark.parametrize('atlas_id,parc_id,region_id', density_profile_param)
def test_single_receptor_profile(atlas_id,parc_id,region_id):
    atlas:Atlas = siibra.atlases[atlas_id]
    parc:Parcellation = atlas.parcellations[parc_id]
    r = parc.decode_region(region_id)
    receptor_feature = siibra.get_features(r, 'receptor')
    assert receptor_feature

queries = siibra.features.FeatureQuery.queries('receptor')

# assert only one type of receptor query is returned
assert len(queries) == 1

receptor_query_obj = queries[0]
recept_dist: List[ReceptorDistribution] = [rd for rd in receptor_query_obj.features]

def _test_profiles(receptor: ReceptorDistribution):

    dict_profiles = JSONEncoder.encode(receptor.profiles, nested=True)
    for key, value in dict_profiles.items():
        assert isinstance(key, str)
        DensityProfile.SiibraSerializationSchema(**value)
        
def _test_finterprint(receptor: ReceptorDistribution):
    dict_fingerprint = JSONEncoder.encode(receptor.fingerprint, nested=True)
    DensityFingerprint.SiibraSerializationSchema(**dict_fingerprint)

def _test_whole(receptor: ReceptorDistribution):
    dict_receptor = JSONEncoder.encode(receptor, nested=True)
    ReceptorDistribution.SiibraSerializationSchema(**dict_receptor)

@pytest.mark.parametrize('receptor', recept_dist)
def test_all_receptor(receptor: ReceptorDistribution):
    _test_profiles(receptor)
    _test_finterprint(receptor)
    _test_whole(receptor)