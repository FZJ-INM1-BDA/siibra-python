from dataclasses import dataclass, field, replace
from typing import Union, Tuple, List
import pandas as pd

from .assignment import find, qualify as collection_qualify
from .qualification import Qualification
from ..atlases import Region
from ..commons import logger
from ..concepts import Attribute
from ..concepts import AttributeCollection
from ..concepts.query_parameter import QueryParam
from ..concepts.feature import Feature
from ..concepts.atlas_elements import AtlasElement
from ..descriptions import Modality
from ..descriptions.modality import vocab as modality_types
from ..exceptions import UnregisteredAttrCompException, InvalidAttrCompException
from ..locations import DataClsLocation


@dataclass
class QueryCursor:
    concept: Union[AtlasElement, DataClsLocation]
    modality: Union[Modality, str]
    additional_attributes: QueryParam = field(default_factory=QueryParam)
    filters: Tuple[Attribute, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.wanted_modality not in modality_types:
            logger.warning(
                f"Modality {self.wanted_modality.value} has not been registered."
                "You might want to check the spelling."
            )

    @property
    def wanted_modality(self):
        if isinstance(self.modality, str):
            modality_obj = modality_types[self.modality]
            logger.info(f"str input {self.modality!r} parsed as {modality_obj}")
            return modality_obj
        if isinstance(self.modality, Modality):
            return self.modality
        raise TypeError(
            f"QueryCursor.modality must be str or Modality, but is {type(self.modality)}"
        )

    @property
    def direct_query_param(self):
        if isinstance(self.concept, DataClsLocation):
            return QueryParam(attributes=[self.concept])
        if isinstance(self.concept, AtlasElement):
            return self.concept
        raise TypeError(
            f"QueryCursor.concept must be DataClsLocation or AtlasElement, but is {type(self.concept)}"
        )

    @property
    def indirect_query_params(self):
        if isinstance(self.direct_query_param, Region):
            from ..dataitems import Image

            # TODO 5632ae5fee97ee65871ba84a3a8f78e5a132cfc4 broke 
            # find_regional_maps(), reenable once it is fixed
            return QueryParam()
            all_maps = self.direct_query_param.find_regional_maps()
            return QueryParam(
                attributes=(attr
                            for map in all_maps
                            for attr in map._find(Image))
            )

        return QueryParam()
    
    @property
    def filter_query_param(self):
        if len(self.filters) == 0:
            return None
        return QueryParam(attributes=self.filters)

    @property
    def query_param(self):
        return QueryParam.merge(
            self.direct_query_param,
            self.indirect_query_params,
            self.additional_attributes,
        )

    @property
    def first_pass_param(self):
        modality_query_param = QueryParam(attributes=[self.wanted_modality])
        return QueryParam.merge(
            modality_query_param, self.additional_attributes, self.indirect_query_params
        )

    def exec(self):
        return [
            feat
            for feat in find(self.first_pass_param, Feature)
            if self.query_param.match(feat)
            and (self.filter_query_param.match(feat) if self.filter_query_param else True)
        ]

    def exec_explain(self, fully=False):
        for feat in self.exec():
            try:
                qualificationiter = collection_qualify(self.query_param, feat)

                if fully:
                    qualifications = list(qualificationiter)
                    if qualifications:
                        yield feat, ", and ".join(
                            [QueryCursor.explain(*q) for q in qualifications]
                        ), qualifications
                    continue

                qualification = next(qualificationiter)
                yield feat, QueryCursor.explain(*qualification), [qualification]
                continue
            except InvalidAttrCompException:
                continue
            except UnregisteredAttrCompException:
                continue
            except StopIteration:
                continue

    @staticmethod
    def explain(attra: Attribute, attrb: Attribute, quality: Qualification):
        return f"{str(attra)} {quality.verb} {str(attrb)}"

    def __iter__(self):
        yield from self.exec()

    def __len__(self):
        return len(self.exec())

    @property
    def facets(self):
        return pd.DataFrame(
            [
                {"key": attr.key, "value": attr.value}
                for f in self.exec()
                for attr in f.facets
            ]
        )
    
    def filter(self, attributes: List[Attribute]):
        return replace(self, filters=self.filters + tuple(attributes))
    
    def filter_by_facets(self, facet_dict=None, **kwargs):
        from ..descriptions import Facet
        if not facet_dict:
            facet_dict = {}
        assert isinstance(facet_dict, dict), f"The positional argument, if supplied, must be a dictionary. But instead is {facet_dict}"

        facets = [
            Facet(key=key, value=value)
            for key, value in {**facet_dict, **kwargs}.items()
        ]
        return self.filter(facets)

