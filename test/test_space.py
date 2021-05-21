import unittest

from siibra.space import Space
from siibra.atlas import REGISTRY


class TestSpaces(unittest.TestCase):

    space_id = 'space_id'
    name = 'space name'
    url = 'space_url'
    ziptarget = 'space_zip_target'
    template = 'temp_file'
    ttype = 'nii'

    json_space_with_zip = {
        '@id': 'space1/minds/core/referencespace/v1.0.0',
        'name': name,
        'shortName': name,
        'templateUrl': url,
        'templateFile': ziptarget,
        'templateType': ttype
    }

    json_space_without_zip = {
        '@id': 'space1/minds/core/referencespace/v1.0.0',
        'name': name,
        'shortName': name,
        'templateUrl': url,
        'templateType': ttype
    }

    def test_space_init(self):
        space = Space(self.space_id, self.name, self.url, self.ziptarget)
        self.assertIsNotNone(space)

    def test_space_from_json_with_zip(self):
        space = Space.from_json(self.json_space_with_zip)
        self.assertEqual(
            str(space),
            self.name
        )
        self.assertEqual(space.type,self.ttype)
        self.assertIsNotNone(space.ziptarget)

    def test_space_from_json_without_zip(self):
        space = Space.from_json(self.json_space_without_zip)
        self.assertEqual(
            str(space),
            self.name
        )
        self.assertIsNone(space.ziptarget)

    def test_space_registry(self):
        spaces = REGISTRY.MULTILEVEL_HUMAN_ATLAS.spaces
        self.assertEqual(len(spaces), 3)


if __name__ == "__main__":
    unittest.main()
