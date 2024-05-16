from dataclasses import dataclass

from .attribute_collection import AttributeCollection

@dataclass
class QueryParam(AttributeCollection):
    schema: str = "siibra/attrCln/queryParam"

    def split_attrs(self):
        for attr in self.attributes:
            yield QueryParam(attributes=[attr])

