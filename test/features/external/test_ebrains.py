import unittest
from unittest.mock import PropertyMock, patch
from siibra.features.dataset.ebrains import EbrainsDataFeature, feature


class TestEbrains(unittest.TestCase):
    @staticmethod
    def get_instance(dataset_id=None):
        return EbrainsDataFeature(
            dataset_id=dataset_id,
            anchor=None,
            name="foo bar",
        )

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_id_uses_ebrains_id(self):
        with patch.object(
            feature.Feature, "id", new_callable=PropertyMock
        ) as mock_feature_id:
            mock_feature_id.return_value = "1"

            ebrains_ds = TestEbrains.get_instance(dataset_id="1223-44")
            self.assertEqual(ebrains_ds.id, "1223-44")
            mock_feature_id.assert_not_called()
