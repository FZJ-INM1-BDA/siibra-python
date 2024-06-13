from dataclasses import dataclass

from .attribute_collection import AttributeCollection


@dataclass
class Feature(AttributeCollection):
    schema: str = "siibra/concepts/feature/v0.2"

    @property
    def aggregate_by(self):
        from ..descriptions import AggregateBy

        return [
            *self._find(AggregateBy),
            *[aggr for attr in self.attributes for aggr in attr.aggregate_by],
        ]

    @property
    def locations(self):
        from ..locations import DataClsLocation
        return self._find(DataClsLocation)

    @property
    def data(self):
        from ..dataitems import Tabular

        return [d.get_data() for d in self._find(Tabular)]

    def plot(self, *args, **kwargs):
        from ..dataitems import Tabular

        return [d.plot(*args, **kwargs) for d in self._find(Tabular)]
