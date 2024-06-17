from dataclasses import dataclass

from .base import Description


@dataclass
class RegionSpec(Description):
    schema = "siibra/attr/desc/regionspec/v0.1"
    parcellation_id: str = None
    value: str = None

    def __str__(self) -> str:
        return (
            f"RegionSpec<{self.value!r}, parcellation_id='{self.parcellation_id[-5:]}'>"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(f"{self.parcellation_id}{self.value}")

    def decode(self):
        from ..assignment import find
        from ..concepts import QueryParam
        from ..atlases import Region

        return find(QueryParam(attributes=[self]), Region)
