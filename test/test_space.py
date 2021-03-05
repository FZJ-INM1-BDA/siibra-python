import unittest

from brainscapes.space import Space


class TestAtlas(unittest.TestCase):

    id = 'space_id'
    name = 'space name'
    url = 'space_url'
    ziptarget = 'space_zip_target'

    def test_space_init(self):
        space = Space(self.id, self.name, self.url, self.ziptarget)


if __name__ == "__main__":
    unittest.main()
