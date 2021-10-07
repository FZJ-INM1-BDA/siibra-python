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

from ..commons import logger

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
        tmpdir = os.environ["SIIBRA_CACHEDIR"] = \
            tempfile.mkdtemp(prefix="siibra-cache-")
        logger.warning(
            f"Siibra created a temporary cache directory at {tmpdir}, as "
            f"the requested folder ({folder}) was not usable. "
            "Please consider to set the SIIBRA_CACHEDIR environment variable "
            "to a suitable directory.")
        return tmpdir


class Cache:

    _instance = None
    folder = user_cache_dir(".".join(__name__.split(".")[:-1]), "")

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
            if "SIIBRA_CACHEDIR" in os.environ:
                cls.folder = os.environ["SIIBRA_CACHEDIR"]
            cls.folder = assert_folder(cls.folder)
            cls._instance = cls.__new__(cls)
        return cls._instance

    def clear(self):
        import shutil

        logger.info(f"Clearing siibra cache at {self.folder}")
        shutil.rmtree(self.folder)
        self.folder = assert_folder(self.folder)

    def build_filename(self, str_rep, suffix=None):
        hashfile = os.path.join(
            self.folder,
            str(hashlib.sha256(str_rep.encode("ascii")).hexdigest())
        )
        if suffix is None:
            return hashfile
        else:
            return hashfile + "." + suffix


CACHE = Cache.instance()
