import unittest
from unittest.mock import patch, PropertyMock, MagicMock, call
from parameterized import parameterized

import siibra
from siibra.configuration.factory import Factory
from siibra import _space, _parcellation
from siibra.core.atlas import Atlas
from siibra.commons import Species
from siibra.core.region import Region
from itertools import product, repeat

Space = _space.Space
Parcellation = _parcellation.Parcellation

human_atlas_json = {
    "@id": "juelich/iav/atlas/v1.0.0/1",
    "@type": "juelich/iav/atlas/v1.0.0",
    "name": "Multilevel Human Atlas",
    "species": "homo sapiens",
    "ebrains": {"minds/core/species/v1.0.0": "0ea4e6ba-2681-4f7d-9fa9-49b915caaac9"},
    "order": 1,
    "spaces": [
        "minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2",
        "minds/core/referencespace/v1.0.0/7f39f7be-445b-47c0-9791-e971c0b6d992",
        "minds/core/referencespace/v1.0.0/a1655b99-82f1-420f-a3c2-fe80fd4c8588",
        "minds/core/referencespace/v1.0.0/tmp-fsaverage",
    ],
    "parcellations": [
        "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579-290",
        "minds/core/parcellationatlas/v1.0.0/94c1125b-b87e-45e4-901c-00daee7f2579",
        "juelich/iav/atlas/v1.0.0/79cbeaa4ee96d5d3dfe2876e9f74b3dc3d3ffb84304fb9b965b1776563a1069c",
        "juelich/iav/atlas/v1.0.0/5",
        "juelich/iav/atlas/v1.0.0/6",
        "juelich/iav/atlas/v1.0.0/4",
        "juelich/iav/atlas/v1.0.0/3",
        "minds/core/parcellationatlas/v1.0.0/d80fbab2-ce7f-4901-a3a2-3c8ef8a3b721",
        "minds/core/parcellationatlas/v1.0.0/73f41e04-b7ee-4301-a828-4b298ad05ab8",
        "minds/core/parcellationatlas/v1.0.0/141d510f-0342-4f94-ace7-c97d5f160235",
        "minds/core/parcellationatlas/v1.0.0/63b5794f-79a4-4464-8dc1-b32e170f3d16",
        "minds/core/parcellationatlas/v1.0.0/12fca5c5-b02c-46ce-ab9f-f12babf4c7e1",
        "https://identifiers.org/neurovault.image:23262",
        "https://doi.org/10.1016/j.jneumeth.2020.108983/mni152",
    ],
}


class MockObj:
    def __init__(self, name=None):
        self.name = name


class MockParc:
    def __init__(self, is_newest_version=True) -> None:
        self.is_newest_version = is_newest_version
        self.find = MagicMock()


class TestAtlas(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        pass

    @classmethod
    def setUpClass(cls):
        cls.atlas = Factory.build_atlas(human_atlas_json)

    def test_species(self):
        self.assertEqual(self.atlas.species, Species.HOMO_SAPIENS)

    def test_id(self):
        self.assertEqual(self.atlas.id, human_atlas_json.get("@id"))

    def test_name(self):
        self.assertEqual(self.atlas.name, human_atlas_json.get("name"))

    def test_spaces(self):
        mocked_spaces = []
        with patch.object(
            Space, "registry", return_value=mocked_spaces
        ) as registry_getitem:
            with patch.object(
                siibra.commons.InstanceTable, "__init__", return_value=None
            ) as mock_init_method:
                _ = self.atlas.spaces
                mock_init_method.assert_called_once_with(
                    elements={}, matchfunc=Space.match
                )
                registry_getitem.assert_called_once_with()

    def test_parcellations(self):
        mocked_parcellations = []
        with patch.object(
            Parcellation, "registry", return_value=mocked_parcellations
        ) as registry_getitem:
            with patch.object(
                siibra.commons.InstanceTable, "__init__", return_value=None
            ) as mock_init_method:
                _ = self.atlas.parcellations
                mock_init_method.assert_called_once_with(
                    elements={}, matchfunc=Parcellation.match
                )
                registry_getitem.assert_called_once_with()

    @parameterized.expand(
        [
            ("spaces", "spaces", "get_space"),
            ("parcellations", "parcellations", "get_parcellation"),
        ]
    )
    def test_get_attr_no_arg(self, property: str, key: str, fn_name: str):
        mocked_return_dict = {_id: MockObj(_id) for _id in human_atlas_json.get(key)}
        with patch(
            f"siibra.core.atlas.Atlas.{property}", new_callable=PropertyMock
        ) as mocked_property:
            mocked_property.return_value = mocked_return_dict

            obj = getattr(self.atlas, fn_name)()
            self.assertEqual(obj.name, list(mocked_return_dict.keys())[0])

    get_space_arg = MockObj()
    get_parc_arg = MockObj()

    @parameterized.expand([(get_parc_arg, get_space_arg)])
    def test_get_map(self, get_parc_arg, get_space_arg):
        space_mock = MockObj()
        parcellation_mock = MockObj()

        with patch.object(
            Atlas, "get_space", return_value=space_mock
        ) as get_space_mock:
            with patch.object(
                Atlas, "get_parcellation", return_value=parcellation_mock
            ) as get_parcellation_mock:
                get_map_return_value = MockObj()

                parcellation_mock.get_map = MagicMock()
                parcellation_mock.get_map.return_value = get_map_return_value

                maptype_arg = MockObj()

                map_return_obj = self.atlas.get_map(
                    get_space_arg, get_parc_arg, maptype_arg
                )
                assert get_map_return_value is map_return_obj

                get_space_mock.assert_called_once_with(get_space_arg)
                get_parcellation_mock.assert_called_once_with(get_parc_arg)
                parcellation_mock.get_map.assert_called_once_with(
                    space=space_mock, maptype=maptype_arg
                )

    get_region_arg = MockObj()

    @parameterized.expand([(get_region_arg, None), (get_region_arg, get_parc_arg)])
    def test_get_region(self, get_region_arg, get_parc_arg):
        parcellation_mock = MockObj()
        parcellation_mock.get_region = MagicMock()
        got_region = MockObj()
        parcellation_mock.get_region.return_value = got_region

        with patch.object(
            Atlas, "get_parcellation", return_value=parcellation_mock
        ) as get_parcellation_mock:
            got_region_return = self.atlas.get_region(get_region_arg, get_parc_arg)
            assert got_region_return is got_region
            get_parcellation_mock.assert_called_once_with(get_parc_arg)
            parcellation_mock.get_region.assert_called_once_with(get_region_arg)

    @parameterized.expand(
        [
            (space_arg, fn_name, mock_space_fn_name, arg, kwarg)
            for space_arg in [get_space_arg, None]
            for fn_name, mock_space_fn_name, args, kwargs in [
                (
                    "get_template",
                    "get_template",
                    [[]],
                    [{"variant": None}, {"variant": MockObj()}],
                ),
                ("get_voi", "get_bounding_box", [(MockObj(), MockObj())], [{}]),
            ]
            for arg in args
            for kwarg in kwargs
        ],
        skip_on_empty=True,
    )
    def test_get_template(self, space_arg, fn_name, mock_space_fn_name, arg, kwarg):
        space_mock = MockObj()

        fn_mock = MagicMock()
        fn_mock_return = MockObj()
        fn_mock.return_value = fn_mock_return
        setattr(space_mock, mock_space_fn_name, fn_mock)

        with patch.object(
            Atlas, "get_space", return_value=space_mock
        ) as get_space_mock:
            return_val = getattr(self.atlas, fn_name)(space_arg, *arg, **kwarg)
            assert return_val is fn_mock_return
            get_space_mock.assert_called_once_with(space_arg)
            fn_mock.assert_called_once_with(*arg, **kwarg)

    @parameterized.expand(
        product(
            ["str-input", 3, Region("hello world")], product([True, False], repeat=3)
        )
    )
    def test_find_regions(self, regionspec, bool_flags):
        all_versions, filter_children, _ = bool_flags
        with patch.object(Parcellation, "get_instance") as get_instance_mock:
            parc1 = MockParc(True)
            parc1.find.return_value = [MockObj(), MockObj(), MockObj()]
            parc2 = MockParc(False)
            parc2.find.return_value = [MockObj(), MockObj(), MockObj()]
            parc3 = MockParc(True)
            parc3.find.return_value = [MockObj(), MockObj(), MockObj()]
            parc4 = MockParc(True)
            parc4.find.return_value = []

            get_instance_mock.side_effect = [parc1, parc2, parc3, *repeat(parc4, 35)]

            actual_result = self.atlas.find_regions(
                regionspec, all_versions, filter_children
            )

            get_instance_mock.assert_has_calls(
                [call(pid) for pid in human_atlas_json.get("parcellations")]
            )
            for p in [parc1, parc2, parc3]:
                if all_versions or p.is_newest_version:
                    p.find.assert_called_once_with(
                        regionspec, filter_children=filter_children
                    )

            self.assertTrue(isinstance(p, MockObj) for p in actual_result)
            flattened = [p for res in actual_result for p in [res]]
            for parc in [parc1, parc2, parc3]:
                for reg in parc.find.return_value:
                    self.assertTrue(
                        (reg in flattened) is (all_versions or parc.is_newest_version)
                    )


if __name__ == "__main__":
    unittest.main()
