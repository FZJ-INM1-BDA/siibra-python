from dataclasses import dataclass

from ..concepts import attribute
from .base import Description

@dataclass
class RegionSpec(Description):
    schema = "siibra/attr/desc/regionspec"
    value: str = None

    # @staticmethod
    # @_m.cache
    # def Matches(regionspec: str, parcellation: str, region: str) -> bool:
    #     found_region = _parcellation.Parcellation.registry()[parcellation].get_region(
    #         region
    #     )
    #     return found_region.matches(regionspec)

    # def matches(self, first_arg=None, *args, region=None, parcellation=None, **kwargs):
    #     if isinstance(first_arg, _region.Region):
    #         region = first_arg
    #     if isinstance(region, _region.Region):
    #         return RegionSpecAttribute.Matches(
    #             self.name, region.parcellation.name, region.name
    #         )
    #     if isinstance(region, str):
    #         assert (
    #             parcellation
    #         ), "If region is supplied as a string, parcellation must be defined!"
    #         assert isinstance(
    #             parcellation, (str, _parcellation.Parcellation)
    #         ), "parcellation must be of type str or Parcellation"
    #         if isinstance(parcellation, _parcellation.Parcellation):
    #             parcellation = parcellation.name
    #         return RegionSpecAttribute.Matches(self.name, parcellation, region)
    #     return super().matches(first_arg, *args, **kwargs)

