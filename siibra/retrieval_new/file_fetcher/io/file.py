from io import FileIO
import os

from .base import PartialReader


class PartialFileReader(PartialReader):
    def __init__(self, filepath: str) -> None:
        super().__init__()

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
