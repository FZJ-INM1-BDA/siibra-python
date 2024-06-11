from dataclasses import dataclass, field
from typing import Union, Tuple
import pandas as pd

from .assignment import get, match as collection_match, qualify as collection_qualify
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
from ..exceptions import UnregisteredAttrCompException
from ..locations import DataClsLocation


@dataclass
class QueryCursor:
    concept: Union[AtlasElement, DataClsLocation]
    modality: Union[Modality, str]
    additional_attributes: QueryParam = field(default_factory=QueryParam)
    filters: Tuple[AttributeCollection, ...] = field(default_factory=tuple)

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
            if (
                self.direct_query_param.name
                != "Area hOc1 (V1, 17, CalcS) - left hemisphere"
            ):
                return QueryParam()
            from ..dataitems import Image

            return QueryParam(
                attributes=[
                    Image(
                        format="nii",
                        space_id="minds/core/referencespace/v1.0.0/dafcffc5-4826-4bf1-8ff6-46b8a31ff8e2",
                        url="https://data-proxy.ebrains.eu/api/v1/buckets/reference-atlas-data/temp/JulichBrainAtlas_3.1/probabilistic-maps_PMs_206-areas/Area-hOc1/Area-hOc1_rh_MNI152.nii.gz",
                    )
                ]
            )

        return QueryParam()

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
            for feat in get(self.first_pass_param, Feature)
            if collection_match(self.query_param, feat)
        ]

    def exec_explain(self, fully=False):
        for feat in get(self.first_pass_param, Feature):
            try:
                print("got feat, trying to qualify")
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

            except UnregisteredAttrCompException as e:
                print("exc???", e)
                continue
            except StopIteration:
                print("couodn't do it")
                continue

    @staticmethod
    def explain(attra: Attribute, attrb: Attribute, quality: Qualification):
        return f"{str(attra)} {quality.verb} {str(attrb)}"

    def __iter__(self):
        yield from self.exec()

    def __len__(self):
        return len(self.exec())

    def aggregate_by(self):
        return pd.DataFrame(
            [
                {"key": attr.key, "value": attr.value}
                for f in self.exec()
                for attr in f.aggregate_by
            ]
        )
