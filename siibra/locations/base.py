from dataclasses import dataclass

from ..concepts.attribute import Attribute

@dataclass
class Location(Attribute):
    schema = "siibra/attr/loc"
    space_id: str = None
