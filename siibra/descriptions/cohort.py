from dataclasses import dataclass

from .base import Description


@dataclass
class Cohort(Description):
    schema = "siibra/attr/desc/cohort/v0.1"
    value: str = None
