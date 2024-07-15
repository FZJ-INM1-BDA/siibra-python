from dataclasses import dataclass

from .base import Description


@dataclass
class Facet(Description):
    schema = "siibra/attr/desc/facet/v0.1"
    key: str = None
    value: str = None

    def __str__(self) -> str:
        return f"{self.key}={self.value}"
