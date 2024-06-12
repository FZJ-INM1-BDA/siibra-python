from dataclasses import dataclass

from .base import Description


@dataclass
class SpeciesSpec(Description):
    schema = "siibra/attr/desc/speciesspec/v0.1"
    value: str = None