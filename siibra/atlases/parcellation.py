from dataclasses import dataclass
from typing import Tuple

from ..atlases import region


@dataclass(init=False)
class Parcellation(region.Region):
    schema: str = "siibra/attrCln/atlasEl/parcellation"
    parent = None

    def __post_init__(self):
        super().__post_init__()
        assert self.parent is None
