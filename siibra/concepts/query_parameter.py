from dataclasses import dataclass
from itertools import product

from .attribute_collection import AttributeCollection
from ..exceptions import UnregisteredAttrCompException, InvalidAttrCompException

@dataclass
class QueryParam(AttributeCollection):
    schema: str = "siibra/attrCln/queryParam"
    and_flag: bool = False
    or_flag: bool = True

    def match(self, ac: AttributeCollection) -> bool:
        from ..assignment.attribute_qualification import is_qualifiable, qualify

        self_attrs = [attr for attr in self.attributes if is_qualifiable(type(attr))]
        other_attrs = [attr for attr in ac.attributes if is_qualifiable(type(attr))]
        
        if self.and_flag:
            for attra, attrb in product(self_attrs, other_attrs):
                try:
                    if qualify(attra, attrb) is None:
                        return False
                except UnregisteredAttrCompException:
                    continue
                except InvalidAttrCompException:
                    return False
            return True

        if self.or_flag:
            for attra, attrb in product(self_attrs, other_attrs):
                try:
                    if qualify(attra, attrb):
                        return True
                except UnregisteredAttrCompException:
                    continue
                except InvalidAttrCompException:
                    continue
            return False

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
