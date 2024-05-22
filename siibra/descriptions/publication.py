from dataclasses import dataclass

from .base import Description
from . import Url, Doi


@dataclass
class Publication(Description):
    schema = "siibra/attr/desc/publication/v0.1"
    citation: str = None
    doi: Doi = None
    url: Url = None
