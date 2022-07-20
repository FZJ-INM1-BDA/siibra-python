from unittest import TestCase, mock, main as run_test
from siibra.core.datasets import EbrainsKgV3DatasetVersion, _EbrainsKgV3Base, EbrainsKgV3Dataset

DATASET_VERSION_TYPE = "https://openminds.ebrains.eu/core/DatasetVersion"
DATASET_VERSION_ID = "fadcd2cb-9e8b-4e01-9777-f4d4df8f1ebc"

DATASET_TYPE = ["https://openminds.ebrains.eu/core/Dataset"]
DATASTE_ID = "https://kg.ebrains.eu/api/instances/82f91c95-6799-485a-ab9a-010c75f9e790"

class TestEbrainsKgV3DatasetVersion(TestCase):
    
    def test_lazy_on_init(self):
        with mock.patch.object(_EbrainsKgV3Base, '_query') as mock_query:
            EbrainsKgV3DatasetVersion({
                '@id': DATASET_VERSION_ID,
                '@type': DATASET_VERSION_TYPE
            })
            assert not mock_query.called

    def test_on_try_desc_called(self):
        with mock.patch.object(_EbrainsKgV3Base, '_query') as mock_query:
            EbrainsKgV3DatasetVersion({
                '@id': DATASET_VERSION_ID,
                '@type': DATASET_VERSION_TYPE
            }).description
            assert mock_query.called
    
    def test_return_desc_if_exists(self):
        with mock.patch.object(EbrainsKgV3Dataset, '_from_json') as mock_parent_json:
            with mock.patch.object(_EbrainsKgV3Base, '_query') as mock_query:
                mock_query.return_value = {
                    'description': 'foo-bar'
                }
                EbrainsKgV3DatasetVersion({
                    '@id': DATASET_VERSION_ID,
                    '@type': DATASET_VERSION_TYPE
                }).description
                assert not mock_parent_json.called

    def test_fallback_to_parent_if_null_desc(self):
        with mock.patch.object(EbrainsKgV3Dataset, '_from_json') as mock_parent_json:
            with mock.patch.object(_EbrainsKgV3Base, '_query') as mock_query:
                mock_query.return_value = {
                    'description': '',
                    'belongsTo': [{
                        "type": DATASET_TYPE,
                        "id": DATASTE_ID
                    }]
                }
                EbrainsKgV3DatasetVersion({
                    '@id': DATASET_VERSION_ID,
                    '@type': DATASET_VERSION_TYPE
                }).description
                assert mock_parent_json.called


if __name__ == "__main__":
    run_test()