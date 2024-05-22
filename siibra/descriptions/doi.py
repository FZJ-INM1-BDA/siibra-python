from dataclasses import dataclass

from .base import Description


@dataclass
class Doi(Description):
    schema = "siibra/attr/desc/doi/v0.1"
    value: str = None
