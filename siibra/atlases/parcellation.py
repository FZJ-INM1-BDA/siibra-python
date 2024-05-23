from dataclasses import dataclass

from ..atlases import region


@dataclass(init=False)
class Parcellation(region.Region):
    schema: str = "siibra/atlases/parcellation/v0.1"
