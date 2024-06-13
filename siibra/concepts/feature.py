from dataclasses import dataclass

from .attribute_collection import AttributeCollection


X_CNMTX_INDEX_BY = "x-siibra/matrix/index/by"
X_CNMTX_INDEX_VALUE = "x-siibra/matrix/index/value"


@dataclass
class Feature(AttributeCollection):
    schema: str = "siibra/concepts/feature/v0.2"

    def get_filters(self):
        return [
            {"connectivity_index": attr.extra[X_CNMTX_INDEX_VALUE]}
            for attr in self.attributes
            if X_CNMTX_INDEX_BY in attr.extra and X_CNMTX_INDEX_VALUE in attr.extra
        ]

    def get_attributes(self, *args, connectivity_index=None, **kwargs):
        if connectivity_index:
            return [
                attr
                for attr in self.attributes
                if attr.extra.get(X_CNMTX_INDEX_VALUE) == connectivity_index
            ]
        return self.attributes

    @property
    def aggregate_by(self):
        from ..descriptions import AggregateBy

        return self._find(AggregateBy)

    @property
    def data(self):
        from ..dataitems import Tabular

        return [d.get_data() for d in self._find(Tabular)]

    def plot(self, *args, **kwargs):
        from ..dataitems import Tabular

        return [d.plot(*args, **kwargs) for d in self._find(Tabular)]
