from dataclasses import dataclass

from .base import Description


@dataclass
class RegionSpec(Description):
    schema = "siibra/attr/desc/regionspec/v0.1"
    parcellation_id: str = None

    def __hash__(self) -> int:
        return hash(f"{self.parcellation_id}{self.value}")

    def decode(self):
        from ..assignment import find
        from ..concepts import QueryParam
        from ..atlases import Region

        return find(QueryParam(attributes=[self]), Region)
