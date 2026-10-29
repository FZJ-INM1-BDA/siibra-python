import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue
from unittest.mock import MagicMock

import pytest

from siibra.retrieval import cache as cache_module
from siibra.retrieval.cache import CACHE, WarmupLevel, Warmup, WarmupRegException

dummy_child1 = MagicMock()
dummy_child2 = MagicMock()

dummy1 = MagicMock(return_value=[dummy_child1, dummy_child2])
dummy2 = MagicMock()

dummy_data1 = MagicMock()
dummy_data2 = MagicMock()


# This fixture temporarily clears production warmup fns
# and restores them after test finishes
@pytest.fixture(scope="session")
def clearwarmup_cache():
    warmup_fns = Warmup._warmup_fns
    Warmup._warmup_fns = []
    yield
    Warmup._warmup_fns = warmup_fns


@pytest.fixture
def register_dummy1(clearwarmup_cache):
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.INSTANCE)(dummy1)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy1)
    dummy1.reset_mock()

    dummy_child1.reset_mock()
    dummy_child2.reset_mock()


@pytest.fixture
def register_dummy2(clearwarmup_cache):
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.INSTANCE)(dummy2)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy2)
    dummy2.reset_mock()


@pytest.fixture
def register_dummy_data1(clearwarmup_cache):
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.DATA)(dummy_data1)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy_data1)
    dummy_data1.reset_mock()


@pytest.fixture
def register_dummy_data2(clearwarmup_cache):
    wrapped_fn = Warmup.register_warmup_fn(WarmupLevel.DATA)(dummy_data2)
    yield wrapped_fn
    Warmup.deregister_warmup_fn(dummy_data2)
    dummy_data2.reset_mock()


@pytest.fixture
def register_all(
    register_dummy1, register_dummy2, register_dummy_data1, register_dummy_data2
):
    yield


def test_register(register_dummy1):
    assert Warmup.is_registered(dummy1)


def test_deregister_original(register_dummy1):
    Warmup.deregister_warmup_fn(dummy1)
    assert not Warmup.is_registered(dummy1)


def test_deregister_wrapped(register_dummy1):
    Warmup.deregister_warmup_fn(register_dummy1)
    assert not Warmup.is_registered(dummy1)


def test_register_multiple():
    wrapped = Warmup.register_warmup_fn()(dummy1)

    with pytest.raises(WarmupRegException):
        Warmup.register_warmup_fn()(dummy1)

    with pytest.raises(WarmupRegException):
        Warmup.register_warmup_fn()(wrapped)

    Warmup.deregister_warmup_fn(dummy1)


def test_register_as_factory():
    wrapped = Warmup.register_warmup_fn(is_factory=True)(dummy1)
    Warmup.warmup()
    dummy1.assert_called_once()
    dummy_child1.assert_called_once()
    dummy_child2.assert_called_once()

    Warmup.deregister_warmup_fn(wrapped)
    dummy1.reset_mock()
    dummy_child1.reset_mock()
    dummy_child2.reset_mock()


def test_register_not_called(register_all):
    dummy1.assert_not_called()
    dummy2.assert_not_called()


def test_register_warmup_called(register_all):
    Warmup.warmup()
    dummy1.assert_called_once()
    dummy2.assert_called_once()

    dummy_data1.assert_not_called()
    dummy_data2.assert_not_called()


def test_register_warmup_called_level(register_all):
    Warmup.warmup(WarmupLevel.INSTANCE)

    dummy1.assert_called_once()
    dummy2.assert_called_once()

    dummy_data1.assert_not_called()
    dummy_data2.assert_not_called()


def test_register_warmup_called_level_high(register_all):
    Warmup.warmup(WarmupLevel.DATA)

    dummy1.assert_called_once()
    dummy2.assert_called_once()

    dummy_data1.assert_called_once()
    dummy_data2.assert_called_once()


SLEEP_TIME = 1


def _sleep_for_x():
    time.sleep(SLEEP_TIME)


def _timed_warmup(queue):
    # Register in the worker itself: processes started with "forkserver" (the
    # default on Linux since Python 3.14) or "spawn" (macOS, Windows) do not
    # inherit warmup functions registered at runtime in the parent process.
    if not Warmup.is_registered(_sleep_for_x):
        Warmup.register_warmup_fn(WarmupLevel.TEST)(_sleep_for_x)
    start = time.time()
    Warmup.warmup(WarmupLevel.TEST)
    queue.put(time.time() - start)


def test_write_lock():
    # Two explicit processes rather than a pool, so each runs one warmup, and
    # timing inside them, so process start-up time can't make the test pass.
    queue = Queue()
    workers = [Process(target=_timed_warmup, args=(queue,)) for _ in range(2)]
    for w in workers:
        w.start()
    durations = sorted(queue.get(timeout=30) for _ in workers)
    for w in workers:
        w.join(timeout=30)

    assert durations[0] >= SLEEP_TIME * 0.9, "warmup function did not run"
    assert durations[1] >= SLEEP_TIME * 1.8, "Expect second call to block"


# --- Cache.fill ---------------------------------------------------------------

def _slow_create(path, content="data", counter=None):
    if counter is not None:
        with open(counter, "a") as f:
            f.write("x")
    time.sleep(0.2)  # widen the race window
    with open(path, "w") as f:
        f.write(content)


def test_fill_creates_once_across_threads(tmp_path):
    target = str(tmp_path / "file")
    calls = []

    def create(path):
        calls.append(path)
        _slow_create(path)

    barrier = threading.Barrier(8)

    def fill(_):
        barrier.wait()
        return CACHE.fill(target, create)

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(fill, range(8)))

    assert results == [target] * 8
    assert len(calls) == 1
    assert open(target).read() == "data"
    assert sorted(os.listdir(tmp_path)) == ["file", "file.lock"]


def _fill_in_process(target, counter):
    CACHE.fill(target, lambda path: _slow_create(path, counter=counter))


def test_fill_creates_once_across_processes(tmp_path):
    target, counter = str(tmp_path / "file"), str(tmp_path / "counter")
    workers = [Process(target=_fill_in_process, args=(target, counter)) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)
    assert all(w.exitcode == 0 for w in workers)
    assert open(counter).read() == "x", "create should run in one process only"
    assert open(target).read() == "data"


def test_fill_existing_target_takes_no_lock(tmp_path, monkeypatch):
    # fill() runs on every cached lookup (e.g. per configuration file), so the
    # common case must stay a plain existence check.
    target = tmp_path / "file"
    target.write_text("cached")

    def no_lock(*args, **kwargs):
        raise AssertionError("an existing target should not take the lock")

    monkeypatch.setattr(cache_module, "Lock", no_lock)
    create = MagicMock()
    assert CACHE.fill(str(target), create) == str(target)
    create.assert_not_called()


def test_fill_failure_leaves_nothing_behind(tmp_path):
    target = str(tmp_path / "file")

    def broken(path):
        with open(path, "w") as f:
            f.write("partial")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError):
        CACHE.fill(target, broken)
    assert sorted(os.listdir(tmp_path)) == ["file.lock"]  # no target, no temp file
    # the lock was released and the next call succeeds
    CACHE.fill(target, lambda path: _slow_create(path, "retry"))
    assert open(target).read() == "retry"


def test_fill_replace(tmp_path):
    target = tmp_path / "file"
    target.write_text("old")
    CACHE.fill(str(target), lambda path: _slow_create(path, "new"), replace=True)
    assert target.read_text() == "new"


def test_fill_folder(tmp_path):
    target = str(tmp_path / "folder")

    def create(path):
        os.makedirs(os.path.join(path, "sub"))
        open(os.path.join(path, "sub", "f.json"), "w").close()

    CACHE.fill(target, create)
    assert os.path.isfile(os.path.join(target, "sub", "f.json"))
    assert sorted(os.listdir(tmp_path)) == ["folder", "folder.lock"]


# --- Cache.contains -------------------------------------------------------------

@pytest.fixture
def cache_folder(tmp_path, monkeypatch):
    folder = tmp_path / "cache"
    folder.mkdir()
    monkeypatch.setattr(CACHE, "folder", str(folder))
    return folder


def test_contains(cache_folder, tmp_path):
    assert CACHE.contains(str(cache_folder / "abc123.json"))
    assert CACHE.contains(str(cache_folder / "sub" / "file"))
    assert not CACHE.contains(str(tmp_path / "my_scan.tif"))
    # a sibling folder that merely shares the prefix
    assert not CACHE.contains(str(tmp_path / "cache2" / "file"))
    # ".." leading out of the cache
    assert not CACHE.contains(str(cache_folder / ".." / "my_scan.tif"))


def test_contains_symlink_out_of_cache(cache_folder, tmp_path):
    user_file = tmp_path / "my_scan.tif"
    user_file.write_text("data")
    link = cache_folder / "link.tif"
    try:
        os.symlink(user_file, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported here")
    assert not CACHE.contains(str(link))
