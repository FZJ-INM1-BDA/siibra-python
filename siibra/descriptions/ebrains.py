from dataclasses import dataclass
from typing import Dict

from ..concepts import attribute
from .base import Description

@dataclass
class EbrainsRef(Description):
    schema = "siibra/attr/desc/ebrains/v0.1"
    ids: Dict[str, str] = None
