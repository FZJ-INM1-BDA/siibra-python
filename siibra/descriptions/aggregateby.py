from dataclasses import dataclass

from .base import Description


@dataclass
class AggregateBy(Description):
    schema = "siibra/attr/desc/aggregateby/v0.1"
    key: str = None
    value: str = None
