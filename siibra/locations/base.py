from dataclasses import dataclass

from ..concepts.attribute import Attribute


@dataclass
class Location(Attribute):
    schema = "siibra/attr/loc"
    space_id: str = None


# static methods

# between two locations

# union
# intersection

# instance method

# warp(target_space_id)
# transform(affine)
