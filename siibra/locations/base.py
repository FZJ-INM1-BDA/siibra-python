from dataclasses import dataclass

from ..concepts.attribute import Attribute
from ..commons_new.iterable import assert_ooo


@dataclass
class Location(Attribute):
    schema = "siibra/attr/loc"
    space_id: str = None

    @property
    def space(self):
        if self.space_id is None:
            return None

        from ..factory import iter_collection
        from ..atlases import Space

        return assert_ooo(
            [space for space in iter_collection(Space) if space.ID == self.space_id]
        )


# static methods

# between two locations

# union
# intersection

# instance method

# warp(target_space_id)
# transform(affine)
