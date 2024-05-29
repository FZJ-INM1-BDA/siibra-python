from dataclasses import dataclass

from .base import Description


@dataclass
class Paradigm(Description):
    schema = "siibra/attr/desc/paradigm/v0.1"
    value: str = None
