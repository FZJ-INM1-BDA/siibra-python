import requests
from filelock import FileLock as Lock
from pathlib import Path
import os

from .base import PartialReader
from .file import PartialFileReader
from ....cache import CACHE
from ....commons import siibra_tqdm
from ....commons import logger

PROGRESS_BAR_THRESHOLD = 2e8
BLOCK_SIZE = 1024


class PartialHttpReader(PartialReader):

    def __init__(self, url: str) -> None:
        super().__init__()
        self.url = url
        self.sess = requests.Session()

        # content = self.probe(0, 50)
        # assert len(content) == 50, f"{len(content)=}"

    def open(self):
        self.sess = requests.Session()

    def close(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None

    def probe(self, offset: int, count: int) -> bytes:
        headers = {
            "Range": f"bytes={offset}-{offset+count-1}"
        }
        resp = self.sess.get(self.url, headers=headers)
        resp.raise_for_status()

        return bytes(resp.content)

    @staticmethod
    def Warmup(url: str):
        filename = CACHE.build_filename(url, suffix=".bin")

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
        return filename

    def warmup(self):
        return PartialFileReader.warmup(self.url)
