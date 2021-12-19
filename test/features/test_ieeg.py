import unittest
from siibra.features.ieeg import IEEG_SessionQuery
import siibra
from siibra.core import Atlas

class TestIEEGSessionQuery(unittest.TestCase):
    colin_spc_query = None
    mni_spc_query = None
    no_spc_query = None

    hoc1_right = None

    @classmethod
    def setUpClass(cls):
        cls.colin_spc_query = IEEG_SessionQuery(space=siibra.spaces['colin'])
        cls.mni_spc_query = IEEG_SessionQuery(space=siibra.spaces['mni152'])
        cls.no_spc_query = IEEG_SessionQuery()
        
        atlas: Atlas = siibra.atlases['human']
        cls.hoc1_right = atlas.get_region('hoc1 right', parcellation='2 9')

    def test_no_spc_returns_all(self):
        result = self.no_spc_query.execute(self.hoc1_right)
        assert len(result) > 0

    def test_colin_return_none(self):
        result = self.colin_spc_query.execute(self.hoc1_right)
        assert len(result) == 0

    def test_mni_returns_all(self):
        result = self.mni_spc_query.execute(self.hoc1_right)
        assert len(result) > 0
