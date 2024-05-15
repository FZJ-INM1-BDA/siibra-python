from dataclasses import dataclass, field
from typing import Dict
from joblib import Memory

from .base import Attribute

from ...core import region as _region, parcellation as _parcellation
from ...locations import Location, Point, PointSet


def all_words_in_string(s1: str, s2: str):
    """returns true if all words from string s1 are found in  string s2."""
    return all(w in s2.lower() for w in s1.lower())


@dataclass
class MetaAttribute(Attribute):
    schema = "siibra/attr/meta"



@dataclass
class PointAttribute(MetaAttribute):
    schema = "siibra/attr/meta/point"
    space_id: str = None
    coordinate: list[float] = field(default_factory=list)

    def matches(self, first_arg=None, *args, location: Location = None, **kwargs):

        if isinstance(first_arg, Location):
            location = first_arg
        if location and location.intersects(
            Point(self.coordinate, space=self.space_id)
        ):
            return True
        return super().matches(first_arg, *args, **kwargs)


@dataclass
class PointSetAttribute(MetaAttribute):
    schema = "siibra/attr/meta/pointset"
    space_id: str = None
    coordinates: list[list[float]] = field(default_factory=list)

    def matches(self, first_arg=None, *args, location: Location = None, **kwargs):

        if isinstance(first_arg, Location):
            location = first_arg
        if location and location.intersects(
            PointSet(self.coordinates, space=self.space_id)
        ):
            return True
        return super().matches(first_arg, *args, **kwargs)


@dataclass
class PolylineDataAttribute(MetaAttribute):
    schema: str = "siibra/attr/meta/polyline"
    closed: bool = False
    space_id: str = None
    coordinates: list[list[float]] = field(default_factory=list)



__all__ = [T.__name__ for T in MetaAttribute.__subclasses__()]
