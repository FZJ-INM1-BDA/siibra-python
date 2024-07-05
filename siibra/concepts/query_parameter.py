from dataclasses import dataclass
from itertools import product

from .attribute_collection import AttributeCollection


@dataclass
class QueryParam(AttributeCollection):
    schema: str = "siibra/attrCln/queryParam"
    and_flag: bool = False
    or_flag: bool = True

    def qualify(self, ac: AttributeCollection):
        from ..assignment.attribute_qualification import simple_qualify

        if self.and_flag:
            return list(
                simple_qualify(attra, attrb)
                for attra, attrb in product(self.attributes, ac.attributes)
            )
        if self.or_flag:
            _iter = (
                simple_qualify(attra, attrb)
                for attra, attrb in product(self.attributes, ac.attributes)
                if simple_qualify(attra, attrb)
            )
            try:
                value = next(_iter)
                if value:
                    return [value]
            except StopIteration:
                return []
        raise RuntimeError("either and_flag or or_flag needs to be set. Both are unset")

    def match(self, ac: AttributeCollection) -> bool:
        from ..assignment.attribute_qualification import simple_qualify

        if self.and_flag:
            return all(
                (
                    simple_qualify(attra, attrb) is not None
                    for attra, attrb in product(self.attributes, ac.attributes)
                )
            )
        if self.or_flag:
            return any(
                (
                    simple_qualify(attra, attrb) is not None
                    for attra, attrb in product(self.attributes, ac.attributes)
                )
            )
        raise RuntimeError("either and_flag or or_flag needs to be set. Both are unset")

    def split_attrs(self):
        for attr in self.attributes:
            yield QueryParam(attributes=[attr])

    @staticmethod
    def merge(*qps: AttributeCollection):
        attributes = tuple()
        for qp in qps:
            attributes += tuple(qp.attributes)
        return QueryParam(attributes=attributes)
