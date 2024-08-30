# Copyright 2018-2024
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

from io import FileIO
import os

from .base import PartialReader


class PartialFileReader(PartialReader):
    def __init__(self, filepath: str) -> None:
        super().__init__(filepath)

        # TODO fix
        # __new__ and __init__ is hard
        if not filepath.startswith("http"):
            self.filepath = filepath
        self.fp: FileIO = None

    def open(self):
        self.fp = open(self.filepath, "rb")

    def close(self):
        if self.fp is not None:
            self.fp.close()
            self.fp = None

    def probe(self, offset: int, count: int) -> bytes:
        self.fp.seek(offset)
        return self.fp.read(count)

    def warmup(self):
        return None

    def get_size(self):
        close_at_end = False
        if self.fp is None:
            self.open()
            close_at_end = True
        size = os.path.getsize(self.fp)
        if close_at_end:
            self.close()
        return size
