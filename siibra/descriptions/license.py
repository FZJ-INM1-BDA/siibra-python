from dataclasses import dataclass

from .base import Description


@dataclass
class License(Description):
    schema = "siibra/attr/desc/license/v0.1"
    value: str = None
    text: str = None
