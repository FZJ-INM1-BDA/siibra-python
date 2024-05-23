from dataclasses import dataclass

from .base import Description


@dataclass
class Publication(Description):
    schema = "siibra/attr/desc/publication/v0.1"
    citation: str = None
    doi: str = None
    url: str = None
