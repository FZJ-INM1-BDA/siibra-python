from dataclasses import dataclass

from .base import Description


@dataclass
class ID(Description):
    schema = "siibra/attr/desc/id/v0.1"
    value: str = None
