from dataclasses import dataclass
from typing import Dict, List, Union

from .base import Description


@dataclass
class EbrainsRef(Description):
    schema = "siibra/attr/desc/ebrains/v0.1"
    ids: Dict[str, Union[str, List[str]]] = None
