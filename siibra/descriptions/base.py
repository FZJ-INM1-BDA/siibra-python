from dataclasses import dataclass

from ..concepts.attribute import Attribute

@dataclass
class Description(Attribute):
    schema = "siibra/attr/desc"
