from dataclasses import dataclass, field
from typing import TypedDict
from ..concepts.attribute import Attribute


class Archive(TypedDict):
    label: int = None
    file: str = None
    format: str = None


@dataclass
class Data(Attribute):
    schema: str = "siibra/attr/data"
    key: str = None
    archive_options: Archive = field(default_factory=Archive)
