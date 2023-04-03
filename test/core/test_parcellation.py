import unittest
import pytest
from siibra.core.parcellation import Parcellation, ParcellationVersion, MapType, MapIndex
from siibra.core.region import Region
from siibra.commons import Species
from uuid import uuid4
from parameterized import parameterized
from unittest.mock import patch, MagicMock
import inspect
from typing import Tuple, Union, NamedTuple
from itertools import product, starmap

correct_json = {
    'name': 'foobar',
    'collectionName': 'foobar-collection',
    '@prev': 'foobar-prev',
    '@next': 'foobar-next',
    'deprecated': False,
}


class DummySpace:
    def matches(self):
        raise NotImplementedError


class DummyParcellation:
    def __init__(self, children) -> None:
        self.children = children
        self.parent = None
        for c in children:
            c.parent = self
        self.find = MagicMock()


class DummyMap:
    def __init__(self, space_returns=True, parcellation_returns=True, maptype=MapType.LABELLED) -> None:
        self.space = DummySpace()
        self.space.matches = MagicMock()
        self.space.matches.return_value = space_returns

        self.parcellation = DummySpace()
        self.parcellation.matches = MagicMock()
        self.parcellation.matches.return_value = parcellation_returns

        self.maptype = maptype
        self.name = ""


class TestParcellationVersion(unittest.TestCase):

    @staticmethod
    def get_instance():
        return ParcellationVersion(
            parcellation=None,
            name=correct_json.get("name"),
            collection=correct_json.get("collectionName"),
            prev_id=correct_json.get("@prev"),
            next_id=correct_json.get("@next"),
            deprecated=correct_json.get("deprecated")
        )

    @classmethod
    def setUpClass(cls):
        cls.parc_version = TestParcellationVersion.get_instance()

    def test_attr(self):
        self.assertTrue(self.parc_version.deprecated == correct_json['deprecated'])
        self.assertTrue(self.parc_version.name == correct_json['name'])
        self.assertTrue(self.parc_version.collection == correct_json['collectionName'])
        self.assertTrue(self.parc_version.prev_id == correct_json['@prev'])
        self.assertTrue(self.parc_version.next_id == correct_json['@next'])


INCORRECT_MAP_TYPE = "incorrect_type"


class InputMap(NamedTuple):
    input: Union[str, MapType]
    alias: MapType


class MapSPMatch(NamedTuple):
    space_match: bool
    parc_match: bool


class MapConfig(NamedTuple):
    map_type: Union[str, MapType]
    sp_match: MapSPMatch


MAP_CONFIG_TYPE = Tuple[MapConfig, MapConfig]


class TestParcellation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.parc = Parcellation(
            identifier=str(uuid4()),
            name="test parc fullname",
            regions=[],
            shortname="test parc shortname",
            description="test parc desc",
            version=TestParcellationVersion.get_instance(),
            modality="test parc modality",
            species=Species.HOMO_SAPIENS,
        )

    # Because of the numerous dependencies of get_map, the parametrized.expand generates a product of most possible scenarios
    @parameterized.expand(
        product(
            # input, space
            [None, "space arg", DummySpace()],
            # input, maptype
            [InputMap(None, MapType.LABELLED), InputMap("labelled", MapType.LABELLED), InputMap(MapType.LABELLED, MapType.LABELLED), InputMap(INCORRECT_MAP_TYPE, None)],
            # volume configuration
            product(
                starmap(MapConfig, product(
                    # maptype of the maps in the registry
                    [MapType.LABELLED, MapType.STATISTICAL],
                    starmap(
                        # whether map.{space,parcellation}.matches should return True/False
                        MapSPMatch,
                        product(
                            [True, False],
                            repeat=2,  # mock result of map.{space,parcellation}.matches
                        )
                    )
                )),
                repeat=2,  # get 2 maps for each
            )
        )
    )
    def test_get_map(self, space, maptype_input: InputMap, vol_spec: Tuple[MapConfig, MapConfig]):
        from siibra.volumes import Map
        ExpectedException = None
        expected_return_idx = None

        for idx, vol_s in enumerate(vol_spec):
            if (
                vol_s.map_type == maptype_input.alias
                and vol_s.sp_match.space_match
                and vol_s.sp_match.parc_match
            ):
                expected_return_idx = idx
                break

        if maptype_input.alias is None:
            ExpectedException = KeyError

        with patch.object(Map, 'registry') as map_registry_mock:
            registry_return = [DummyMap(config.sp_match.space_match, config.sp_match.parc_match, config.map_type) for config in list(vol_spec)]

            map_registry_mock.return_value = registry_return
            if inspect.isclass(ExpectedException) and issubclass(ExpectedException, Exception):
                with self.assertRaises(ExpectedException):
                    map = self.parc.get_map(space, maptype_input.input)
                return

            args = []
            if space is not None:
                args.append(space)
            if maptype_input.input is not None:
                args.append(maptype_input.input)

            map = self.parc.get_map(*args)
            map_registry_mock.assert_called_once()
            if expected_return_idx is not None:
                self.assertIs(map, registry_return[expected_return_idx])
            else:
                self.assertIsNone(map)

    @parameterized.expand([
        (True,),
        (False,),
    ])
    def test_find_regions(self, parents_only):
        Parcellation._CACHED_REGION_SEARCHES = {}
        with patch.object(Parcellation, 'registry') as parcellation_registry_mock:
            parc1 = DummyParcellation([])
            parc2 = DummyParcellation([])
            parc3 = DummyParcellation([parc1, parc2])

            for p in [parc1, parc2, parc3]:
                p.find.return_value = [p]
            parcellation_registry_mock.return_value = [parc1, parc2, parc3]

            result = Parcellation._find_regions("fooz", parents_only)

            parcellation_registry_mock.assert_called_once()
            for p in [parc1, parc2, parc3]:
                p.find.assert_called_once_with(regionspec="fooz")
            self.assertEqual(result, [
                parc3
            ] if parents_only else [parc1, parc2, parc3])

    @parameterized.expand(
        product(
            ["region_spec", 4, MapIndex(0, 1), Region("test-region")],
            [True, False],
            [True, False]
        )
    )
    @unittest.skip
    def test_get_region(self, regionspec, find_topmost, build_group):
        pass

# all_parcs = [p for p in parcellations]

# @pytest.mark.parametrize('parc', all_parcs)
# def test_parc_to_model(parc: Parcellation):
#     model = parc.to_model()
#     import re
#     assert re.match(r"^[\w/\-.:]+$", model.id), f"model_id should only contain [\w/\-.:]+, but is instead {model.id}"

# all_parc_models = [parc.to_model() for parc in all_parcs]
# all_regions = [
#     (pev, bav)
#     for model in all_parc_models
#     for bav in model.brain_atlas_versions
#     for pev in bav.has_terminology_version.has_entity_version]

# @pytest.mark.parametrize('pev_id_dict,bav', all_regions)
# def test_parc_regions(pev_id_dict,bav):
#     filtered_pev = [pev for pev in bav.has_terminology_version.has_entity_version if pev.get("@id") == pev_id_dict.get("@id")]
#     if len(filtered_pev) == 1:
#         assert True
#     else:
#         pytest.xfail(f"PEV with id {pev_id_dict.get('@id')} has multiple instances.")

# fetch_map_param = [
#     ('rat', 'waxholm', 'v4', 'labelled')
# ]

# @pytest.mark.parametrize('atlas_id,space_id,parc_id,map_type', fetch_map_param)
# def test_should_be_able_to_fetch_map(atlas_id,space_id,parc_id,map_type):

#     atlas = siibra.atlases[atlas_id]
#     space = atlas.spaces[space_id]
#     parc = atlas.parcellations[parc_id]

#     parc.get_map(space, map_type)


parc_has_ebrains_doi = [
    ("human", "julich brain 2.9")
]
