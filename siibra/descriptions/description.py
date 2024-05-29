from dataclasses import dataclass

from .base import Description


@dataclass
class TextDescription(Description):
    schema = "siibra/attr/desc/description/v0.1"
    value: str = None
