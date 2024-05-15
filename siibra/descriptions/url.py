from dataclasses import dataclass

from ..concepts import attribute
from .base import Description

@dataclass
class Url(Description):
    schema = "siibra/attr/desc/url"
    value: str = None
