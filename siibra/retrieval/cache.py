# Copyright 2018-2026
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Maintaining and handling caching files on disk."""

import hashlib
import os
from appdirs import user_cache_dir
import tempfile
from functools import wraps
from enum import Enum
from typing import Callable, List, NamedTuple, Union
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filelock import FileLock as Lock

from ..commons import (
    logger,
    SIIBRA_CACHEDIR,
    SIIBRA_CACHE_SIZE_GIB,
    SKIP_CACHEINIT_MAINTENANCE,
    siibra_tqdm,
)
from ..exceptions import WarmupRegException


JOBLIB_FOLDERNAME = "joblib"


def assert_folder(folder):
    # make sure the folder exists and is writable, then return it.
    # If it cannot be written, create and return
    # a temporary folder.
    try:
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if not os.access(folder, os.W_OK):
            raise OSError
        return folder
    except OSError:
        # cannot write to requested directory, create a temporary one.
        tmpdir = tempfile.mkdtemp(prefix="siibra-cache-")
        logger.warning(
            f"Siibra created a temporary cache directory at {tmpdir}, as "
            f"the requested folder ({folder}) was not usable. "
            "Please consider to set the SIIBRA_CACHEDIR environment variable "
            "to a suitable directory.")
        return tmpdir


class Cache:

    _instance = None
    folder = user_cache_dir(".".join(__name__.split(".")[:-1]), "")
    SIZE_GIB = SIIBRA_CACHE_SIZE_GIB  # maintenance will delete old files to stay below this limit
    _EXCLUDED_FROM_MAINTENANCE = frozenset({JOBLIB_FOLDERNAME})

    def __init__(self):
        raise RuntimeError(
            "Call instance() to access "
            f"{self.__class__.__name__}")

    @classmethod
    def instance(cls):
        """
        Return an instance of the siibra cache. Create folder if needed.
        """
        if cls._instance is None:
            if SIIBRA_CACHEDIR:
                cls.folder = SIIBRA_CACHEDIR
            cls.folder = assert_folder(cls.folder)
            cls._instance = cls.__new__(cls)
            if SKIP_CACHEINIT_MAINTENANCE:
                logger.debug("Will not run maintenance on cache as SKIP_CACHE_MAINTENANCE is set to True.")
            else:
                cls._instance.run_maintenance()
        return cls._instance

    def clear(self):
        import shutil

        logger.info(f"Clearing siibra cache at {self.folder}")
        shutil.rmtree(self.folder)
        self.folder = assert_folder(self.folder)

    def run_maintenance(self):
        """
        Shrinks the cache by deleting oldest files first until the total size
        is below cache size (Cache.SIZE_GIB) given in GiB.

        Use `SIIBRA_CACHE_SIZE_GIB` environment variable before import, or
        `siibra.set_cache_size()` to change the cache size.
        """
        # os.scandir: DirEntry.stat() is cached per entry, and on Windows it
        # comes with the directory listing, avoiding one system call per file.
        with os.scandir(self.folder) as it:
            entries = [e for e in it if e.name not in self._EXCLUDED_FROM_MAINTENANCE]
        sentries = sorted(entries, key=lambda e: e.stat().st_atime)

        # determine the first n files that need to be deleted to reach the accepted cache size
        targetsize = sum(e.stat().st_size for e in sentries) / 1024**3
        index = 0
        for index, entry in enumerate(sentries):
            if targetsize <= self.SIZE_GIB:
                break
            targetsize -= entry.stat().st_size / 1024**3

        if index > 0:
            logger.debug(f"Removing the {index + 1} oldest files to keep cache size below {self.SIZE_GIB:.2f} GiB.")
            for entry in sentries[:index + 1]:
                if entry.is_dir(follow_symlinks=False):
                    import shutil
                    shutil.rmtree(entry.path)
                else:
                    os.remove(entry.path)

    @property
    def size(self):
        """Return size of the cache in GiB."""
        with os.scandir(self.folder) as it:
            return sum(e.stat().st_size for e in it) / 1024**3

    def __iter__(self):
        """Iterate all element names in the cache directory."""
        return (os.path.join(self.folder, f) for f in os.listdir(self.folder))

    def build_filename(self, str_rep: str, suffix=None) -> str:
        """
        Generate a filename in the cache.

        Parameters
        ----------
        str_rep: str
            Unique string representation of the item. Will be used to compute a hash.
        suffix: str. Default: None
            Optional file suffix, in order to allow filetype recognition by the name.

        Returns
        -------
            str
        """
        hashfile = os.path.join(
            self.folder, str(hashlib.sha256(str_rep.encode("ascii")).hexdigest())
        )
        if suffix is None:
            return hashfile
        else:
            if suffix.startswith("."):
                return hashfile + suffix
            else:
                return hashfile + "." + suffix


CACHE = Cache.instance()


class WarmupLevel(int, Enum):
    TEST = -1000
    INSTANCE = 1
    DATA = 5


class WarmupParam(NamedTuple):
    level: Union[int, WarmupLevel]
    fn: Callable
    is_factory: bool = False


class Warmup:

    _warmup_fns: List[WarmupParam] = []

    @staticmethod
    def fn_eql(wrapped_fn, original_fn):
        return wrapped_fn is original_fn or wrapped_fn.__wrapped__ is original_fn

    @classmethod
    def is_registered(cls, fn):
        return len([warmup_fn.fn
                    for warmup_fn in cls._warmup_fns
                    if cls.fn_eql(warmup_fn.fn, fn)]) > 0

    @classmethod
    def register_warmup_fn(cls, warmup_level: WarmupLevel = WarmupLevel.INSTANCE, *, is_factory=False):
        def outer(fn):
            if cls.is_registered(fn):
                raise WarmupRegException

            @wraps(fn)
            def inner(*args, **kwargs):
                return fn(*args, **kwargs)

            cls._warmup_fns.append(WarmupParam(warmup_level, inner, is_factory))
            return inner
        return outer

    @classmethod
    def deregister_warmup_fn(cls, original_fn):
        cls._warmup_fns = [
            warmup_fn for warmup_fn in cls._warmup_fns
            if not cls.fn_eql(warmup_fn.fn, original_fn)
        ]

    @classmethod
    def warmup(cls, warmup_level: WarmupLevel = WarmupLevel.INSTANCE, *, max_workers=4):
        all_fns = [warmup for warmup in cls._warmup_fns if warmup.level <= warmup_level]

        def call_fn(fn: WarmupParam):
            return_val = fn.fn()
            if not fn.is_factory:
                return
            for f in return_val:
                f()

        with Lock(CACHE.build_filename("lockfile", ".warmup")):
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for _ in siibra_tqdm(
                    ex.map(
                        call_fn,
                        all_fns
                    ),
                    desc="Warming cache",
                    total=len(all_fns),
                ):
                    ...


try:
    from joblib import Memory
    jobmemory_path = Path(CACHE.folder) / JOBLIB_FOLDERNAME
    jobmemory_path.mkdir(parents=True, exist_ok=True)
    jobmemory = Memory(jobmemory_path, verbose=0)
    cache_user_fn = jobmemory.cache
except ImportError:
    from functools import lru_cache
    cache_user_fn = lru_cache
