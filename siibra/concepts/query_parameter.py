from dataclasses import dataclass

from .attribute_collection import AttributeCollection

@dataclass
class QueryParam(AttributeCollection):
    schema: str = "siibra/attrCln/queryParam"
    pass
