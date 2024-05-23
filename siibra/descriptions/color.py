from dataclasses import dataclass

from .base import Description


@dataclass
class RGBColor(Description):
    schema = "siibra/attr/desc/rgbcolor/v0.1"
    value: str = None
