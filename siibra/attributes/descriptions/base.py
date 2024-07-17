from dataclasses import dataclass

from ...attributes.attribute import Attribute


@dataclass
class Description(Attribute):
    schema = "siibra/attr/desc"
    value: str = None
