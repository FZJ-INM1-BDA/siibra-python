from unittest.mock import patch

from siibra.configuration.factory import Factory
from siibra.retrieval import repositories


def test_dataproxy_repository():
    spec = {"repository": {
        "@type": "siibra/repository/dataproxy/v1.0.0",
        "bucketname": "d-41db823e-7e1b-44c7-9c69-eaa26e226384",
        "folder": "MNI-JulichBrain-3.0",
    }}
    with patch.object(repositories, "BucketApiClient") as client:
        connector = Factory.extract_connector(spec)
    client.assert_not_called()
    assert isinstance(connector, repositories.DataProxyConnector)
    assert connector.bucketname == "d-41db823e-7e1b-44c7-9c69-eaa26e226384"
    assert connector.main_folder == "MNI-JulichBrain-3.0"


def test_dataproxy_repository_without_folder():
    spec = {"repository": {"@type": "siibra/repository/dataproxy/v1.0.0", "bucketname": "some-bucket"}}
    assert Factory.extract_connector(spec).main_folder == ""
