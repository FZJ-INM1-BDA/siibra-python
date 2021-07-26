import unittest
from io import BytesIO
from unittest.mock import patch, MagicMock

from siibra.features import receptors
import tempfile


class TestReceptors(unittest.TestCase):

    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False)

    def test_receptor_symbols(self):
        self.assertEqual(len(receptors.RECEPTOR_SYMBOLS), 16)

    def test_unify_string_list(self):
        unified_list = receptors.unify_stringlist(['a', 'b', 'a', 'c', 'b', 'a'])
        self.assertEqual(unified_list, ['a', 'b', 'a*', 'c', 'b*', 'a**'])

    def test_unify_error_if_not_all_string_items(self):
        with self.assertRaises(AssertionError):
            receptors.unify_stringlist(['a', 'b', 1])
        self.assertTrue('Assertion error is expected')


if __name__ == "__main__":
    unittest.main()
