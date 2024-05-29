from dataclasses import dataclass

from .base import Description


@dataclass
class RegionSpec(Description):
    schema = "siibra/attr/desc/regionspec/v0.1"
    parcellation_id: str = None
    value: str = None
