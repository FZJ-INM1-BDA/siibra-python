from dataclasses import dataclass

from .attribute_collection import AttributeCollection

@dataclass
class AtlasElement(AttributeCollection):
    schema: str = "siibra/attrCln/atlasEl"
    pass
