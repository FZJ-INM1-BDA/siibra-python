from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from siibra.retrieval import repositories
from siibra.retrieval.repositories import DataProxyConnector


def test_creating_a_connector_does_not_look_up_its_bucket():
    with patch.object(repositories, "BucketApiClient") as client:
        connector = DataProxyConnector("d-41db823e-7e1b-44c7-9c69-eaa26e226384", main_folder="MNI-JulichBrain-3.0")
        url = connector._build_url("", "amplitude__mad.txt.gz")
    client.assert_not_called()
    assert url == (
        "https://data-proxy.ebrains.eu/api/v1/buckets/d-41db823e-7e1b-44c7-9c69-eaa26e226384/"
        "MNI-JulichBrain-3.0/amplitude__mad.txt.gz"
    )


def test_listing_looks_up_the_bucket_once():
    bucketname = f"test-{uuid4().hex}"  # listings are cached per bucket
    with patch.object(repositories, "BucketApiClient") as client:
        bucket = client.return_value.buckets.get_bucket.return_value
        bucket.ls.return_value = [SimpleNamespace(name="a/x.tsv"), SimpleNamespace(name="a/y.csv")]
        connector = DataProxyConnector(bucketname)
        assert connector.search_files("a/", ".tsv", recursive=True) == ["a/x.tsv"]
        assert connector.search_files("a/", ".csv", recursive=True) == ["a/y.csv"]
    client.return_value.buckets.get_bucket.assert_called_once_with(bucketname)


def test_listings_of_different_buckets_do_not_mix():
    listings = {}
    for name in ["a", "b"]:
        bucketname = f"test-{name}-{uuid4().hex}"
        with patch.object(repositories, "BucketApiClient") as client:
            client.return_value.buckets.get_bucket.return_value.ls.return_value = [
                SimpleNamespace(name=f"shared/{name}.tsv")
            ]
            listings[name] = DataProxyConnector(bucketname).search_files("shared/", ".tsv", recursive=True)
    assert listings == {"a": ["shared/a.tsv"], "b": ["shared/b.tsv"]}
