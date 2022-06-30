import unittest
import pytest
import siibra
from siibra import parcellations
from siibra.core import Parcellation

class TestParcellationVersion(unittest.TestCase):
    correct_json={
        'name': 'foobar',
        'collectionName': 'foobar-collection',
        '@prev': 'foobar-prev',
        '@next': 'foobar-next',
        'deprecated': False,
    }
    def test__from_json(self):

        ver=siibra.core.parcellation.ParcellationVersion._from_json(self.correct_json)
        self.assertTrue(ver.deprecated == self.correct_json['deprecated'])
        self.assertTrue(ver.name == self.correct_json['name'])
        self.assertTrue(ver.collection == self.correct_json['collectionName'])
        self.assertTrue(ver.prev_id == self.correct_json['@prev'])
        self.assertTrue(ver.next_id == self.correct_json['@next'])
        # TODO test prev/next
        
class TestParcellation(unittest.TestCase):

    correct_json={
        '@id': 'id-foo',
        '@type': 'minds/core/parcellationatlas/v1.0.0',
        'shortName': 'id-foo-shortname',
        'name':'fooparc',
        'regions': []
    }

    correct_json_no_type={
        **correct_json,
        '@type': 'foo-bar'
    }

    def test_from_json_malformed(self):
        self.assertRaises(AssertionError, lambda: Parcellation._from_json(self.correct_json_no_type))
    
    def test_from_json(self):
        parc = Parcellation._from_json(self.correct_json)
        assert isinstance(parc, Parcellation)

    def test_find_regions_ranks_result(self):
        updated_json = {
            **self.correct_json,
            'regions': [{
                'name': 'foo bar',
                'children': [{
                    'name': 'foo'
                }]
            }]
        }
        parc = Parcellation._from_json(updated_json)
        regions = parc.find_regions('foo')
        assert len(regions) == 3
        assert regions[0].name == 'foo'

all_parcs = [p for p in parcellations]

@pytest.mark.parametrize('parc', all_parcs)
def test_parc_to_model(parc: Parcellation):
    model = parc.to_model()
    import re
    assert re.match(r"^[\w/\-.:]+$", model.id), f"model_id should only contain [\w/\-.:]+, but is instead {model.id}"

all_parc_models = [parc.to_model() for parc in all_parcs]
all_regions = [
    (pev, bav)
    for model in all_parc_models
    for bav in model.brain_atlas_versions
    for pev in bav.has_terminology_version.has_entity_version]

@pytest.mark.parametrize('pev_id_dict,bav', all_regions)
def test_parc_regions(pev_id_dict,bav):
    filtered_pev = [pev for pev in bav.has_terminology_version.has_entity_version if pev.get("@id") == pev_id_dict.get("@id")]
    if len(filtered_pev) == 1:
        assert True
    else:
        pytest.xfail(f"PEV with id {pev_id_dict.get('@id')} has multiple instances.")

fetch_map_param = [
    ('rat', 'waxholm', 'v4', 'labelled')
]

@pytest.mark.parametrize('atlas_id,space_id,parc_id,map_type', fetch_map_param)
def test_should_be_able_to_fetch_map(atlas_id,space_id,parc_id,map_type):
    
    atlas = siibra.atlases[atlas_id]
    space = atlas.spaces[space_id]
    parc = atlas.parcellations[parc_id]

    parc.get_map(space, map_type)

parc_has_ebrains_doi = [
    ("human", "julich brain 2.9")
]

@pytest.mark.parametrize('atlas_id,parc_id', parc_has_ebrains_doi)
def test_should_have_ebrains_doi(atlas_id,parc_id):
    
    atlas = siibra.atlases[atlas_id]
    parc = atlas.parcellations[parc_id]
    model = parc.to_model()
    all(
        ver.digital_identifier is not None
        and ver.digital_identifier.get("@type") == "https://openminds.ebrains.eu/core/DOI"
        and ver.digital_identifier.get("@id").startswith("https://doi.org")
        for ver in model.brain_atlas_versions
    )

if __name__ == "__main__":
    unittest.main()
