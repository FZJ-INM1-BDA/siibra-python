from dataclasses import dataclass

from .base import Description

@dataclass
class Name(Description):
    schema = "siibra/attr/desc/name/v0.1"
    value: str = None
