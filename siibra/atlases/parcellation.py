from dataclasses import dataclass

from ..atlases import region


@dataclass(init=False)
class Parcellation(region.Region):
    schema: str = "siibra/atlases/parcellation/v0.1"
    parent = None

    def __post_init__(self):
        super().__post_init__()
        assert self.parent is None
