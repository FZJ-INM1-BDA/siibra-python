from dataclasses import dataclass

from .base import Description


@dataclass
class Url(Description):
    schema = "siibra/attr/desc/url/v0.1"
    text: str = None
