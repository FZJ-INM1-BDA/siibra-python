from dataclasses import dataclass, field
from typing import List, Tuple

from .base import Location
from .point import Pt

@dataclass
class Polyline(Location):
    schema: str = "siibra/attr/loc/polyline"
    closed: bool = False
    points: Tuple[Pt] = field(default_factory=tuple)

