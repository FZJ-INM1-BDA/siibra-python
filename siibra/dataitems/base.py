from dataclasses import dataclass

from ..concepts.attribute import Attribute

@dataclass
class Data(Attribute):
    schema: str = "siibra/attr/data"
