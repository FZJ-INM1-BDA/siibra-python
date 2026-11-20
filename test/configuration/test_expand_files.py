from unittest.mock import patch

import pandas as pd
import pytest

from siibra.configuration.factory import Factory
from siibra.retrieval import repositories
from siibra.retrieval.repositories import DataProxyConnector, LocalFileRepository

FILES = {"sub-1": "sub-1/data.tsv", "sub-2": "sub-2/data.tsv"}


def local(folder, **spec):
    return {"repository": {"@type": "siibra/repository/localfolder/v1.0.0", "folder": str(folder)}, "files": FILES, **spec}


def test_one_entry_per_file_sharing_one_connector(tmp_path):
    entries = Factory.expand_files(local(tmp_path))
    assert [(e.key, e.filename) for e in entries] == list(FILES.items())
    assert isinstance(entries[0].connector, LocalFileRepository)
    assert entries[0].connector is entries[1].connector


def test_dataproxy_files_are_linked_exactly():
    spec = {
        "repository": {"@type": "siibra/repository/dataproxy/v1.0.0", "bucketname": "some-bucket", "folder": "v1"},
        "files": FILES,
    }
    with patch.object(repositories, "BucketApiClient") as client:
        entries = Factory.expand_files(spec)
        urls = [e.connector.get_loader(e.filename, decode_func=e.decode_func).url for e in entries]
    client.assert_not_called()  # nothing is listed
    assert isinstance(entries[0].connector, DataProxyConnector)
    assert urls == [
        "https://data-proxy.ebrains.eu/api/v1/buckets/some-bucket/v1/sub-1/data.tsv",
        "https://data-proxy.ebrains.eu/api/v1/buckets/some-bucket/v1/sub-2/data.tsv",
    ]


@pytest.mark.parametrize("spec, message", [
    ({"files": FILES}, "needs a 'repository'"),
    ({"base_url": "https://example.org/", "files": FILES}, "'base_url' is not supported"),
    ({"repository": {"@type": "siibra/repository/unknown/v1.0.0"}, "files": FILES}, "no connector"),
])
def test_invalid_sources(spec, message):
    with pytest.raises(ValueError, match=message):
        Factory.expand_files(spec)


def test_decoder_comes_from_spec_and_file_name(tmp_path):
    decoder = Factory.expand_files(local(tmp_path, decoder={"delimiter": "\t"}))[0].decode_func
    assert decoder.func is pd.read_csv
    assert decoder.kwargs == {"delimiter": "\t"}


def test_ids_derive_from_spec_id(tmp_path):
    assert [e.id for e in Factory.expand_files(local(tmp_path, **{"@id": "abc"}))] == ["abc--sub-1", "abc--sub-2"]
    assert Factory.expand_files(local(tmp_path))[0].id is None


def test_no_files_no_entries(tmp_path):
    assert Factory.expand_files({**local(tmp_path), "files": {}}) == []
