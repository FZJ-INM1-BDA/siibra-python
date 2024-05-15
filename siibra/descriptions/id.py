from dataclasses import dataclass

from ..concepts import attribute
from .base import Description

@dataclass
class ID(Description):
    schema = "siibra/attr/desc/id"
    uuid: str = None

