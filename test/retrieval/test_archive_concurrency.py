import io
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from siibra.retrieval import cache as cache_module
from siibra.retrieval import repositories
from siibra.retrieval.cache import CACHE


def make_tarball(path, files):
    """Write a tarball laid out like GitHub's: the top-level folder comes first."""
    with tarfile.open(path, "w:gz") as tar:
        for name in ["owner-repo-abc1234", "owner-repo-abc1234/sub"]:
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        for name, content in files:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


class Downloads(list):
    tarball = None


@pytest.fixture
def local_tarball(tmp_path, monkeypatch):
    """Serve a small repository tarball instead of downloading one."""
    downloads = Downloads()
    downloads.tarball = tmp_path / "repo.tar.gz"
    make_tarball(downloads.tarball, [("owner-repo-abc1234/config.json", b"{}"),
                                     ("owner-repo-abc1234/sub/spec.json", b"[]")])

    class FakeRequest:
        def __init__(self, url, *args, **kwargs):
            self.cachefile = str(downloads.tarball)

        def _retrieve(self):
            downloads.append(self.cachefile)
            time.sleep(0.05)  # a real download is slow; widen the race window

    monkeypatch.setattr(repositories, "HttpRequest", FakeRequest)
    monkeypatch.setattr(CACHE, "folder", str(tmp_path / "cache"))
    os.makedirs(CACHE.folder)
    return downloads


def make_connector():
    return repositories.GithubConnector(
        "owner", "repo", reftag="v1", skip_branchtest=True, archive_mode=True
    )


def test_concurrent_archive_fetch(local_tarball):
    """
    siibra.warm_cache() loads the configuration from several threads. With an
    empty cache, all of them used to extract the archive into the same folder,
    and every rename but the first failed with "Directory not empty".
    """
    n = 8
    connectors = [make_connector() for _ in range(n)]
    barrier = threading.Barrier(n)

    def fetch(conn):
        barrier.wait()  # start all threads at the same moment
        conn._archive()
        return str(conn._archive_conn._folder)

    with ThreadPoolExecutor(n) as ex:
        folders = list(ex.map(fetch, connectors))

    assert len(set(folders)) == 1
    assert os.path.isfile(os.path.join(folders[0], "sub", "spec.json"))
    assert len(local_tarball) == 1, "the archive should be downloaded only once"
    leftovers = [f for f in os.listdir(CACHE.folder) if ".tmp-" in f]
    assert leftovers == []


def test_cached_archive_takes_no_lock(local_tarball, monkeypatch):
    """
    _archive() runs once per configuration file loaded, so once the archive is
    extracted it must return without locking (locking cost ~40% of load time).
    """
    conn = make_connector()
    conn._archive()

    def no_lock(*args, **kwargs):
        raise AssertionError("cached archive should not take the file lock")

    monkeypatch.setattr(cache_module, "Lock", no_lock)
    for _ in range(100):
        conn._archive()
    assert len(local_tarball) == 1


@pytest.mark.skipif(not hasattr(tarfile, "data_filter"),
                    reason="tarfile extraction filters need Python >= 3.12 or a security backport")
def test_archive_rejects_paths_outside_cache(local_tarball, tmp_path):
    make_tarball(local_tarball.tarball, [
        ("owner-repo-abc1234/config.json", b"{}"),
        ("owner-repo-abc1234/../../escaped.txt", b"pwned"),
    ])
    with pytest.raises(tarfile.FilterError):
        make_connector()._archive()
    assert not any(tmp_path.rglob("escaped.txt"))
    leftovers = [f for f in os.listdir(CACHE.folder) if ".tmp-" in f]
    assert leftovers == []
