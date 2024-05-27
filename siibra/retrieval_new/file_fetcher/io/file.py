from io import FileIO

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
