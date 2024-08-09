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

import requests
from filelock import FileLock as Lock
from pathlib import Path
import os

from .base import PartialReader
from ....cache import CACHE
from ....commons_new.logger import logger, siibra_tqdm

PROGRESS_BAR_THRESHOLD = 2e8
BLOCK_SIZE = 1024


class PartialHttpReader(PartialReader):

    def __init__(self, url: str) -> None:
        super().__init__(url)
        self.url = url
        self.sess = requests.Session()

    def open(self):
        self.sess = requests.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def probe(self, offset: int, count: int) -> bytes:
        end = str(offset + count - 1)
        if count == -1:
            end = ""
        headers = {"Range": f"bytes={offset}-{end}"}
        resp = self.sess.get(self.url, headers=headers)
        resp.raise_for_status()

        return bytes(resp.content)

    @staticmethod
    def WarmFilename(url: str):
        return CACHE.build_filename(url, suffix=".bin")

    @staticmethod
    def IsWarm(url: str):
        filename = PartialHttpReader.WarmFilename(url)
        return Path(filename).is_file()

    @staticmethod
    def Warmup(url: str):
        if PartialHttpReader.IsWarm(url):
            return
        filename = PartialHttpReader.WarmFilename(url)

        with Lock(filename + ".lock"):
            if Path(filename).is_file():
                return filename
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            size_bytes = int(resp.headers.get("content-length", 0))
            progress_bar = None
            if size_bytes == 0 or size_bytes >= PROGRESS_BAR_THRESHOLD:
                desc = f"Downloading {url}"
                desc += (
                    f" ({size_bytes / 1024**2:.1f} MiB)"
                    if size_bytes > 0
                    else " (Unknown size)"
                )
                progress_bar = siibra_tqdm(
                    total=size_bytes,
                    unit="iB",
                    position=0,
                    leave=True,
                    unit_scale=True,
                    desc=desc,
                )
            tmp_filename = filename + ".tmp"
            with open(tmp_filename, "wb") as fp:
                for data in resp.iter_content(BLOCK_SIZE):
                    if progress_bar is not None:
                        progress_bar.update(len(data))
                    fp.write(data)
                if progress_bar is not None:
                    progress_bar.close()
            logger.info(f"Download {url} completed. Cleaning up ...")
            os.rename(tmp_filename, filename)

    def warmup(self):
        return PartialHttpReader.warmup(self.url)

    def get_size(self) -> int:
        headers = self.get_headers()
        content_length = headers.get("content-length")
        if content_length is None:
            raise NotImplementedError(
                f"{self.url} does not support content-length header."
            )
        return int(content_length)

    def get_headers(self):
        resp = requests.get(self.url, stream=True)
        resp.close()
        return resp.headers
