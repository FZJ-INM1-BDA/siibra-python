from dataclasses import dataclass

from .base import Description


@dataclass
class Version(Description):
    schema = "siibra/attr/desc/version/v0.1"
    prev_id: str = None  # None if there is no previous
    next_id: str = None  # None if there is no next
