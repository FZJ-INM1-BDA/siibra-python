from unittest.mock import patch, MagicMock

from siibra import Parcellation, get_region
from test.core.test_region import TestRegion

def test_get_region():
    class DummyCls:
        def get_region(self):
            raise NotImplementedError
    dummy_obj = DummyCls()
    mock_region = TestRegion.get_instance()
    dummy_obj.get_region = MagicMock(return_value=mock_region)
    with patch.object(Parcellation, 'get_instance', return_value=dummy_obj) as mock_get_instance:
        actual_got_region = get_region("boo-buz", "foo-bar")

        mock_get_instance.assert_called_once_with("boo-buz")
        dummy_obj.get_region.assert_called_once_with("foo-bar")
        assert actual_got_region is mock_region