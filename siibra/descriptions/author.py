from dataclasses import dataclass

from ..concepts import attribute
from .base import Description

@dataclass
class Author(Description):
    schema = "siibra/attr/desc/author"
    name: str = None
    affiliation: str = ""
