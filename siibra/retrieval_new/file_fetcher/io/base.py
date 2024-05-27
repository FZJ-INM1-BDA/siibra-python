from abc import ABC, abstractmethod
import requests

from ....commons import logger

BLOCK_SIZE = 512


class PartialReader(ABC):

    def __new__(cls, path: str):
        from .file import PartialFileReader
        from .http import PartialHttpReader

        if path is None:
            return super().__new__(cls)

        if str(path).startswith("http"):
            resp = requests.get(
                path, headers={"Range": f"bytes=0-{BLOCK_SIZE}"}, stream=True
            )
            resp.raise_for_status()
            total_size = 0
            for data in resp.iter_content(BLOCK_SIZE):
                total_size += len(data)
                if total_size > BLOCK_SIZE:
                    resp.close()
                    logger.warning(
                        f"{path} does not support range requests. Downloading all file first."
                    )
                    filename = PartialHttpReader.Warmup(path)

                    instance = PartialFileReader.__new__(cls, filename)
                    PartialFileReader.__init__(instance, filename)
                    return instance
            instance = object.__new__(PartialHttpReader)
            instance.url = path
            return instance

        instance = object.__new__(PartialFileReader)
        instance.filepath = path
        return instance

    def __init__(self) -> None:
        super().__init__()
        self.marker = 0

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
        raise NotImplementedError

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, et, ev, tb):
        self.close()
