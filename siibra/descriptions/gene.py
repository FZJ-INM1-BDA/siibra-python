from dataclasses import dataclass

from .base import Description

@dataclass
class Gene(Description):
    schema = "siibra/desc/gene/v0.1"
    value: str = None
