from dataclasses import dataclass

from .base import Description


@dataclass
class Version(Description):
    schema = "siibra/attr/desc/version/v0.1"
    value: str = None  # version in string format
    prev_id: str = None  # None if there is no previous
    next_id: str = None  # None if there is no next
