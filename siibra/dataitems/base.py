from dataclasses import dataclass, field
from typing import TypedDict

class Archive(TypedDict):
    label: int = None
    file: str = None

from ..concepts.attribute import Attribute

@dataclass
class Data(Attribute):
    schema: str = "siibra/attr/data"
    key: str = None
    archive_options: Archive = field(default_factory=Archive)
