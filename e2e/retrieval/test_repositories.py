import pytest
from siibra.retrieval.repositories import DataProxyConnector


@pytest.mark.parametrize("input_str, expected_bucketname", [
    ("https://data-proxy.ebrains.eu/api/v1/buckets/test-sept-22/", "test-sept-22"),
    ("https://data-proxy.ebrains.eu/api/v1/buckets/test-sept-22/foo/bar", "test-sept-22"),
    ("https://data-proxy.ebrains.eu/api/v1/buckets/test-sept-22", "test-sept-22"),
    ("https://data-proxy.ebrains.eu/test-sept-22/", "test-sept-22"),
    ("https://data-proxy.ebrains.eu/test-sept-22?prefix=foofoobar%2F", "test-sept-22"),
    ("test-sept-22", "test-sept-22"),
])
def test_inputs(input_str: str, expected_bucketname: str):
    conn = DataProxyConnector(input_str)
    assert conn.bucketname == expected_bucketname


@pytest.mark.parametrize("folder, suffix, item_count_min, item_count_max", [
    ("", ".mp4", 20, 1e10),  # slow first time, because searching root
    ("", ".mp5", 0, 0),  # slow first time, because searching root
    ("foo", "", 5, 1e10),
    ("./foo", "", 5, 1e10),
    ("./foo", ".txt", 5, 1e10),
    ("./foo", ".apple", 0, 0),
])
def test_ls(folder: str, suffix: str, item_count_min: int, item_count_max: int):
    conn = DataProxyConnector("test-sept-22")
    files = conn.search_files(folder, suffix, recursive=True)
    assert item_count_min <= len(files) <= item_count_max


@pytest.mark.parametrize("folder, filename, expected_str", [
    ("./foo", "bus_2.txt", "hello world\n"),
    ("foo", "bus_2.txt", "hello world\n"),
    ("", "foo/bus_2.txt", "hello world\n"),
    ("", "./foo/bus_2.txt", "hello world\n"),
    ("./", "./foo/bus_2.txt", "hello world\n"),
    (".", "./foo/bus_2.txt", "hello world\n"),
])
def test_fetching(folder: str, filename: str, expected_str: str):
    conn = DataProxyConnector("test-sept-22")
    d = conn.get(filename, folder, decode_func=lambda b: b.decode("utf-8"))
    assert d == expected_str
