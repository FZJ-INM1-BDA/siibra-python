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

    @patch('siibra.features.receptors.retrieval.download_file')
    def test_get_bytestream_from_file(self, download_file_mock):
        self.file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        download_file_mock.return_value = self.file.name
        bytestream_result = receptors.get_bytestream_from_file('')
        self.assertIsNotNone(bytestream_result)
        self.assertTrue(isinstance(bytestream_result, BytesIO))

    def test_unify_string_list(self):
        unified_list = receptors.unify_stringlist(['a', 'b', 'a', 'c', 'b', 'a'])
        self.assertEqual(unified_list, ['a', 'b', 'a*', 'c', 'b*', 'a**'])

    def test_unify_error_if_not_all_string_items(self):
        with self.assertRaises(AssertionError):
            receptors.unify_stringlist(['a', 'b', 1])
        self.assertTrue('Assertion error is expected')

    @patch('siibra.features.receptors.get_bytestream_from_file')
    def test_decode_tsv(self, get_bytestream_mock):
        bytestream = MagicMock()
        get_bytestream_mock.return_value = bytestream
        # Data for bytestream return must be defined
        #bytestream.readline.return_value = ['header1', 'header2']
        #bytestream.readlines.return_value = [b'line1.1 \t line1.2', b'line2.1 \t line2.2']
        #receptors.decode_tsv('')


if __name__ == "__main__":
    unittest.main()
