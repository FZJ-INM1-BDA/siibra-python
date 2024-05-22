from dataclasses import dataclass

from .attribute_collection import AttributeCollection


@dataclass
class Feature(AttributeCollection):
    schema: str = "siibra/feature/v0.2"
