from dataclasses import dataclass, field
from itertools import product
from typing import List, Union
import pandas as pd

from .atlas_elements import AtlasElement
from .attribute_collection import AttributeCollection
from ..exceptions import UnregisteredAttrCompException, InvalidAttrCompException
from ..locations import DataClsLocation
from ..descriptions import Modality
from ..descriptions.modality import vocab as modality_types
from ..commons_new.logger import logger

@dataclass
class QueryParamCollection:

    criteria: List[AttributeCollection] = field(default_factory=list)

    # if set, all criteria must be met (default)
    # takes precedence over or_flag
    and_flag = True

    # if set, any criteria is met
    or_flag = False

    @staticmethod
    def from_concept_modality(concept: Union[AtlasElement, DataClsLocation], modality: Union[str, Modality], additional_attribute_collections: List[AttributeCollection]=None):
        concept_attr_collection = None
        if isinstance(concept, DataClsLocation):
            concept_attr_collection = AttributeCollection(attributes=[concept])
        if isinstance(concept, AtlasElement):
            concept_attr_collection = concept
        if concept_attr_collection is None:
            raise ValueError(f"concept needs to be either DataClsLocation or AtlasElement, but is instead {type(concept)}")
        
        modality_attr_collection = None
        if isinstance(modality, str):
            msg = f"str input {modality!r}"
            modality = modality_types[modality]
            logger.info(f"{msg} parsed as {modality}")
        if isinstance(modality, Modality):
            modality_attr_collection = AttributeCollection(attributes=[modality])
        if modality_attr_collection is None:
            raise ValueError(f"modality needs to be either str or Modality, but is instead {type(modality)}")

        return QueryParamCollection(criteria=[concept_attr_collection,
                                              modality_attr_collection,
                                              *(additional_attribute_collections if additional_attribute_collections else [])])

    def match(self, ac: AttributeCollection) -> bool:
        from ..assignment import collection_match
        if self.and_flag:
            return all((collection_match(cri, ac) for cri in self.criteria))
        if self.or_flag:
            return any((collection_match(cri, ac) for cri in self.criteria))
        raise RuntimeError("either and_flag or or_flag must be set. Neither was")

    def exec(self):
        from ..assignment import find
        from .feature import Feature
        return [
            feat
            for feat in find(self, Feature)
        ]
    
    @property
    def facets(self):
        return pd.DataFrame([{"key": facet.key, "value": facet.value}
                             for f in self.exec()
                             for facet in f.facets])

@dataclass
class QueryParam(AttributeCollection):
    schema: str = "siibra/attrCln/queryParam"

    def match(self, ac: AttributeCollection) -> bool:
        from ..assignment.attribute_qualification import is_qualifiable, qualify

        self_attrs = [attr for attr in self.attributes if is_qualifiable(type(attr))]
        other_attrs = [attr for attr in ac.attributes if is_qualifiable(type(attr))]
        
        for attra, attrb in product(self_attrs, other_attrs):
            try:
                if qualify(attra, attrb):
                    return True
            except UnregisteredAttrCompException:
                continue
            except InvalidAttrCompException:
                continue
        return False
    
    def split_attrs(self):
        for attr in self.attributes:
            yield QueryParam(attributes=[attr])

    @staticmethod
    def merge(*qps: AttributeCollection):
        attributes = tuple()
        for qp in qps:
            attributes += tuple(qp.attributes)
        return QueryParam(attributes=attributes)
