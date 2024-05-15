from dataclasses import dataclass

from ..concepts import attribute
from .base import Description


@dataclass
class SpeciesSpec(Description):
    schema = "siibra/attr/desc/speciesspec"
    name: str = None
