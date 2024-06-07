from dataclasses import dataclass
from typing import Union
import pandas as pd

from ..concepts.query_parameter import QueryParam
from ..concepts.feature import Feature
from ..concepts.atlas_elements import AtlasElement
from ..locations import DataClsLocation
from ..descriptions import Modality


@dataclass
class QueryCursor:
    concept: Union[AtlasElement, DataClsLocation]
    modality: Union[Modality, str]

    @property
    def wanted_modality(self):
        if isinstance(self.modality, str):
            return Modality(value=self.modality)
        if isinstance(self.modality, Modality):
            return self.modality
        raise TypeError(
            f"QueryCursor.modality must be str or Modality, but is {type(self.modality)}"
        )

    @property
    def query_param(self):
        if isinstance(self.concept, DataClsLocation):
            return QueryParam(attributes=[self.concept])
        if isinstance(self.concept, AtlasElement):
            return self.concept
        raise TypeError(
            f"QueryCursor.concept must be DataClsLocation or AtlasElement, but is {type(self.concept)}"
        )

    def exec(self):
        from . import get, collection_match

        return [
            feat
            for feat in get(QueryParam(attributes=[self.wanted_modality]), Feature)
            if collection_match(self.query_param, feat)
        ]

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
