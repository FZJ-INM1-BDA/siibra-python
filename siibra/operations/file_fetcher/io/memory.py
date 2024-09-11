from .base import PartialReader


class MemoryPartialReader(PartialReader):
    def __init__(self, b: bytes) -> None:
        self.bytes = b

    def open(self):
        pass

    def close(self):
        pass

    def probe(self, offset: int, count: int) -> bytes:
        return self.bytes[offset : offset + count]

    def get_size(self) -> int:
        return len(self.bytes)
