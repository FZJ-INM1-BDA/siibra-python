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

from abc import ABC, abstractmethod
import io
import requests

from ....commons.logger import logger

BLOCK_SIZE = 512


class PartialReader(io.IOBase, ABC):
    """
    Read a specific file/URL with random access.

    Note
    ----
    At initiation, the class will determine which class to initialize:

    - if path does not start with https://, returns an instance of PartialFileReader,
      which is a proxy to FileIO
    - if path starts with https://
        - if server does not support range request, downloads the file immediately, and
          return PartialFileReader pointing to the newly downloaded file
        - if the server support range requests, returns an instance of PartialHttpReader

    As the download of the unsuitable file happens at initialization time, usage of this
    class should be as lazily as possible.
    """

    def __new__(cls, path: str):
        from .file import PartialFileReader
        from .http import PartialHttpReader

        if path is None:
            return super().__new__(cls)

        # path starts with https://, assume to be a URL
        if str(path).startswith("https://"):
            http_warm_path = PartialHttpReader.WarmFilename(path)

            # If the url is already cached on disk, use file reader
            if PartialHttpReader.IsWarm(path):
                instance = PartialFileReader.__new__(cls, http_warm_path)
                PartialFileReader.__init__(instance, http_warm_path)
                return instance

            # Check server supports range request
            resp = requests.get(
                path, headers={"Range": f"bytes=0-{BLOCK_SIZE-1}"}, stream=True
            )
            resp.raise_for_status()
            total_size = 0
            for data in resp.iter_content(BLOCK_SIZE):
                total_size += len(data)

                # If received size > expected size, server does not support range request
                # call Warmup, and return file reader
                if total_size > BLOCK_SIZE:
                    resp.close()
                    logger.warning(
                        f"{path} does not support range requests. Downloading all file first."
                    )
                    PartialHttpReader.Warmup(path)

                    instance = PartialFileReader.__new__(cls, http_warm_path)
                    PartialFileReader.__init__(instance, http_warm_path)
                    return instance

            # Server supports range request, return partial http request
            instance = io.IOBase.__new__(PartialHttpReader)
            instance.url = path
            size = instance.get_size()
            instance._size = size
            return instance

        # if path does not start with https://, use file reader
        instance = io.IOBase.__new__(PartialFileReader)
        instance.filepath = path
        return instance

    _size = None

    def __init__(self, path: str = None) -> None:
        super().__init__()
        self.marker = 0

    @property
    def size(self):
        if self._size is None:
            self._size = self.get_size()
        return self._size

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence=0):
        if whence == 0:
            self.marker = offset
            return self.marker
        elif whence == 1:
            self.marker += offset
            return self.marker
        elif whence == 2:
            self.marker = self.size + offset
            return self.marker
        else:
            raise RuntimeError(f"whence={whence!r} must be 0, 1, 2")

    def read(self, size=-1):
        start_marker = self.marker
        current_marker = start_marker + size
        self.marker = current_marker
        return self.probe(start_marker, size)

    def tell(self):
        return self.marker

    @abstractmethod
    def warmup(self):
        raise NotImplementedError

    @abstractmethod
    def probe(self, offset: int, count: int) -> bytes:
        """
        Implementation of reading partial buffer.

        Note
        ----
        count can be set to -1, to indicate from offset to the end of buffer.

        Parameters
        ----------
        offset: int
        count: int

        Returns
        -------
        bytes
        """
        raise NotImplementedError

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def get_size(self) -> int:
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, et, ev, tb):
        self.close()
