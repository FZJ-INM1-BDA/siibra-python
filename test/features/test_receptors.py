import unittest
import pytest

from siibra.features.receptors import RECEPTOR_SYMBOLS, unify_stringlist, ReceptorQuery, ReceptorDistribution
import tempfile


class TestReceptors(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    def test_receptor_symbols(self):
        self.assertEqual(len(RECEPTOR_SYMBOLS), 16)

    def test_unify_string_list(self):
        unified_list = unify_stringlist(['a', 'b', 'a', 'c', 'b', 'a'])
        self.assertEqual(unified_list, ['a', 'b', 'a*', 'c', 'b*', 'a**'])

    def test_unify_error_if_not_all_string_items(self):
        with self.assertRaises(AssertionError):
            unify_stringlist(['a', 'b', 1])
        self.assertTrue('Assertion error is expected')


receptor_query = ReceptorQuery()

@pytest.mark.parametrize('receptor', receptor_query.features)
def test_receptor_to_model(receptor: ReceptorDistribution):
    receptor.to_model()

if __name__ == "__main__":
    unittest.main()
