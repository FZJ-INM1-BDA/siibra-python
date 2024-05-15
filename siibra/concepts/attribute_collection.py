from dataclasses import dataclass, field
from typing import List

from .attribute import Attribute

@dataclass
class AttributeCollection:
    schema: str = "siibra/attrCln"
    name: str = None
    attributes: List[Attribute] = field(default_factory=list)
