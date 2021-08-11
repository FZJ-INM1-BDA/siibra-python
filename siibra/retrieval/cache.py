# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

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

from ..commons import logger


class Cache:

    _instance = None
    folder = user_cache_dir(".".join(__name__.split(".")[:-1]), "")

    def __init__(self):
        raise RuntimeError(f"Call instance() to access {self.__class__.__name__}")

    @classmethod
    def instance(cls):
        if cls._instance is None:
            if "SIIBRA_CACHEDIR" in os.environ:
                cls.folder = os.environ["SIIBRA_CACHEDIR"]
            # make sure cachedir exists and is writable
            try:
                if not os.path.isdir(cls.folder):
                    os.makedirs(cls.folder)
                assert os.access(cls.folder, os.W_OK)
                logger.debug(f"Setup cache at {cls.folder}")
            except Exception as e:
                print(str(e))
                raise PermissionError(
                    f"Cannot create cache at {cls.folder}. Please define a writable cache directory in the environment variable SIIBRA_CACHEDIR."
                )
            cls._instance = cls.__new__(cls)
        return cls._instance

    def clear(self):
        import shutil

        logger.info(f"Clearing siibra cache at {self.folder}")
        shutil.rmtree(self.folder)

    def build_filename(self, str_rep, suffix=None):
        hashfile = os.path.join(
            self.folder, str(hashlib.sha256(str_rep.encode("ascii")).hexdigest())
        )
        if suffix is None:
            return hashfile
        else:
            return hashfile + "." + suffix


CACHE = Cache.instance()
