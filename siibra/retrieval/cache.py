# Copyright 2018-2021
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

import hashlib
import os
from appdirs import user_cache_dir
import tempfile

from ..commons import logger, SIIBRA_CACHEDIR


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
    SIZE_GIB = 2  # maintenance will delete old files to stay below this limit

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
            cls._instance.run_maintenance()
        return cls._instance

    def clear(self):
        import shutil

        logger.info(f"Clearing siibra cache at {self.folder}")
        shutil.rmtree(self.folder)
        self.folder = assert_folder(self.folder)

    def run_maintenance(self):
        """ Shrinks the cache by deleting oldest files first until the total size
        is below cache size (Cache.SIZE) given in GiB."""

        # build sorted list of cache files and their os attributes
        files = [os.path.join(self.folder, fname) for fname in os.listdir(self.folder)]
        sfiles = sorted([(fn, os.stat(fn)) for fn in files], key=lambda t: t[1].st_atime)

        # determine the first n files that need to be deleted to reach the accepted cache size
        size_gib = sum(t[1].st_size for t in sfiles) / 1024**3
        targetsize = size_gib
        index = 0
        for index, (fn, st) in enumerate(sfiles):
            if targetsize <= self.SIZE_GIB:
                break
            targetsize -= st.st_size / 1024**3

        if index > 0:
            logger.debug(f"Removing the {index+1} oldest files to keep cache size below {targetsize:.2f} GiB.")
            for fn, st in sfiles[:index + 1]:
                size_gib -= st.st_size / 1024**3
                os.remove(fn)

    @property
    def size(self):
        """ Return size of the cache in GiB. """
        return sum(os.path.getsize(fn) for fn in self) / 1024**3

    def __iter__(self):
        """ Iterate all element names in the cache directory. """
        return (os.path.join(self.folder, f) for f in os.listdir(self.folder))

    def build_filename(self, str_rep: str, suffix=None):
        """Generate a filename in the cache.

        Args:
            str_rep (str): Unique string representation of the item. Will be used to compute a hash.
            suffix (str, optional): Optional file suffix, in order to allow filetype recognition by the name. Defaults to None.

        Returns:
            filename
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
