# Copyright 2018-2024
# Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field, replace
from itertools import product
from typing import List, Union
import pandas as pd

from .atlas_elements import AtlasElement
from ..attributes import AttributeCollection
from ..exceptions import UnregisteredAttrCompException, InvalidAttrCompException
from ..attributes.locations import DataClsLocation
from ..attributes.descriptions import Modality
from ..attributes.descriptions.modality import vocab as modality_types
from ..commons_new.logger import logger


@dataclass
class QueryParamCollection:

    criteria: List[AttributeCollection] = field(default_factory=list)
    filters: List[AttributeCollection] = field(default_factory=list)

    # if set, all criteria must be met (default)
    # takes precedence over or_flag
    and_flag = True

    # if set, any criteria is met
    or_flag = False

    @staticmethod
    def from_concept_modality(
        concept: Union[AtlasElement, DataClsLocation],
        modality: Union[str, Modality],
        additional_attribute_collections: List[AttributeCollection] = None,
    ):
        concept_attr_collection = None
        if isinstance(concept, DataClsLocation):
            concept_attr_collection = AttributeCollection(attributes=[concept])
        if isinstance(concept, AtlasElement):
            concept_attr_collection = concept
        if concept_attr_collection is None:
            raise ValueError(
                f"concept needs to be either DataClsLocation or AtlasElement, but is instead {type(concept)}"
            )

        modality_attr_collection = None
        if isinstance(modality, str):
            msg = f"str input {modality!r}"
            modality = modality_types[modality]
            logger.info(f"{msg} parsed as {modality}")
        if isinstance(modality, Modality):
            modality_attr_collection = AttributeCollection(attributes=[modality])
        if modality_attr_collection is None:
            raise ValueError(
                f"modality needs to be either str or Modality, but is instead {type(modality)}"
            )

        return QueryParamCollection(
            criteria=[
                concept_attr_collection,
                modality_attr_collection,
                *(
                    additional_attribute_collections
                    if additional_attribute_collections
                    else []
                ),
            ]
        )

    def match(self, ac: AttributeCollection) -> bool:
        from ..assignment import collection_match

        # filters are always applied, regardless of flag
        # in python any([]) == False, all([]) == True
        # this is to cach the scenario that filters is an empty list
        if any((not collection_match(f, ac)) for f in self.filters):
            return False

        if self.and_flag:
            return all((collection_match(cri, ac) for cri in self.criteria))
        if self.or_flag:
            return any((collection_match(cri, ac) for cri in self.criteria))
        raise RuntimeError("either and_flag or or_flag must be set. Neither was")

    def exec(self):
        from ..assignment import find
        from .feature import Feature

        return [feat for feat in find(self, Feature)]

    @property
    def facets(self):
        return pd.DataFrame(
            [
                {"key": facet.key, "value": facet.value}
                for f in self.exec()
                for facet in f.facets
            ]
        )

    def filter_by_facets(self, facet_dict=None, **kwargs):
        from ..attributes.descriptions import Facet

        if not facet_dict:
            facet_dict = {}
        assert isinstance(
            facet_dict, dict
        ), f"The positional argument, if supplied, must be a dictionary. But instead is {facet_dict}"

        facets = [
            Facet(key=key, value=value)
            for key, value in {**facet_dict, **kwargs}.items()
        ]
        new_filter = AttributeCollection(attributes=facets)
        return replace(self, filters=[*self.filters, new_filter])


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
